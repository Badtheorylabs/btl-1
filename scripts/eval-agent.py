"""
Agent CLI evaluator for the compression-trained model.

Runs the model in interactive chat mode, batch eval mode, or
benchmark suite mode using the existing eval/ framework.

Usage:
  python scripts/eval-agent.py --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter artifacts/compression-adapter
  python scripts/eval-agent.py --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter artifacts/compression-adapter --interactive
  python scripts/eval-agent.py --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter artifacts/compression-adapter --benchmark --limit 50
  python scripts/eval-agent.py --gguf compression-model.gguf --interactive
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="BTL Agent CLI Evaluator")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Base model name (HF) for loading with adapter")
    model_group.add_argument("--gguf", help="Path to GGUF model file")

    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for benchmark")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--output", default=None, help="Output file for benchmark results")
    return parser.parse_args()


def load_hf_model(base_model: str, adapter_path: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device if device != "auto" else "auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def load_gguf_model(gguf_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading GGUF model: {gguf_path}")
    model = Llama(
        model_path=gguf_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    return model, None


def predict_hf(model, tokenizer, prompt: str, max_tokens: int, temperature: float, device: str):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device if hasattr(model, "device") else device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def predict_gguf(model, prompt: str, max_tokens: int, temperature: float):
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>", "<|im_end|>", "<|end|>"],
    )
    return output["choices"][0]["text"].strip()


def interactive_loop(predict_fn, tokenizer, max_tokens: int, temperature: float):
    print("\n=== BTL Agent CLI Interactive Mode ===")
    print("Type your prompts. Enter 'quit' to exit.\n")

    SYSTEM_PROMPT = (
        "You are a helpful coding agent. You can use tools to solve tasks.\n"
        "Available tools:\n"
        "- read_file(path) - Read file contents\n"
        "- write_file(path, content) - Write content to file\n"
        "- list_files(path) - List directory contents\n"
        "- run_command(command) - Execute shell command\n"
        "- search_code(query, path) - Search codebase\n"
        "Return your answer in JSON format with 'tool' and 'arguments' fields, "
        "or use 'reasoning' for non-tool responses."
    )

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit", "/quit"):
            break

        if not user_input.strip():
            continue

        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\n\nAssistant:"
        start = time.time()
        response = predict_fn(prompt)
        elapsed = time.time() - start

        print(f"\n{response}")
        print(f"\n[Generated in {elapsed:.2f}s]")
        print()


def run_benchmark(predict_fn, tokenizer, limit: int | None, output: str | None):
    project_root = Path(__file__).resolve().parent.parent

    sys.path.insert(0, str(project_root))
    from eval.run_all import run_suite, render_suite

    result = run_suite(predict_fn, project_root, tokenizer=tokenizer, limit=limit)
    print(render_suite(result))

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_suite(result))
        print(f"\nResults saved to {out_path}")


def run_single_prompt(predict_fn, prompt: str, max_tokens: int, temperature: float, output: str | None):
    print(f"Prompt: {prompt}\n")
    start = time.time()
    response = predict_fn(prompt)
    elapsed = time.time() - start

    print(f"Response:\n{response}")
    print(f"\n[Generated in {elapsed:.2f}s]")

    if output:
        result = {"prompt": prompt, "response": response, "elapsed_s": elapsed}
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nResult saved to {out_path}")


def main():
    args = parse_args()

    if args.gguf:
        model = load_gguf_model(args.gguf)
        predict_fn = lambda prompt: predict_gguf(model, prompt, args.max_tokens, args.temperature)
        tokenizer = None
    else:
        model, tokenizer = load_hf_model(args.model, args.adapter, args.device)
        predict_fn = lambda prompt: predict_hf(model, tokenizer, prompt, args.max_tokens, args.temperature, args.device)

    if args.interactive:
        interactive_loop(predict_fn, tokenizer, args.max_tokens, args.temperature)
    elif args.benchmark:
        run_benchmark(predict_fn, tokenizer, args.limit, args.output)
    elif args.prompt:
        run_single_prompt(predict_fn, args.prompt, args.max_tokens, args.temperature, args.output)
    else:
        interactive_loop(predict_fn, tokenizer, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
