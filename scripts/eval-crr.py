"""
Standalone CRR (Compression Ratio Ratio) evaluator.

Measures compression quality of a trained model vs an SFT-only baseline
by computing tool orchestration scores on train and transfer distributions.

Usage:
  python scripts/eval-crr.py --adapter artifacts/compression-adapter
  python scripts/eval-crr.py --adapter artifacts/compression-adapter --baseline-adapter artifacts/sft-baseline
  python scripts/eval-crr.py --gguf compression-model.gguf --limit 100
  python scripts/eval-crr.py --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="CRR Evaluator")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (3-loss model)")
    parser.add_argument("--baseline-adapter", default=None, help="Path to SFT-only baseline adapter")
    parser.add_argument("--gguf", default=None, help="Path to GGUF model file")
    parser.add_argument("--model", default=None, help="Base model name (evaluate untrained model)")
    parser.add_argument("--eval-path", default=None,
                        help=f"Eval dataset path (default: $BTL_EVAL_PATH or data/real-traces/eval.jsonl)")
    parser.add_argument("--train-path", default=None,
                        help=f"Train dataset path (default: $BTL_TRAIN_PATH or data/real-traces/train.jsonl)")
    parser.add_argument("--limit", type=int, default=200, help="Number of eval rows to evaluate")
    parser.add_argument("--max-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--output", default=None, help="Output file for results (JSON)")
    return parser.parse_args()


def load_model(base_model: str | None, adapter_path: str | None, gguf_path: str | None, device: str):
    import torch

    if gguf_path:
        from llama_cpp import Llama
        model = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        tokenizer = None
        is_gguf = True
        return model, tokenizer, is_gguf

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if base_model is None:
        base_model = os.environ.get("BTL_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

    print(f"Loading model: {base_model}")
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
    return model, tokenizer, False


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_model(model, tokenizer, eval_rows, is_gguf, max_len, max_new_tokens, device):
    import torch

    depth_scores = defaultdict(list)
    depth_valid_json = defaultdict(list)
    B = 4

    if is_gguf:
        for r in eval_rows:
            depth = r.get("provenance", {}).get("source_depth", -1)
            msgs = r.get("messages", [])
            prompt_msgs = msgs[:2]

            prompt_parts = []
            for m in prompt_msgs:
                role = m.get("role", "user").upper()
                content = m.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("ASSISTANT:")
            prompt_text = "\n".join(prompt_parts)

            output = model(prompt_text, max_tokens=max_new_tokens, temperature=0.0, stop=["</s>", "<|im_end|>"])
            gen_text = output["choices"][0]["text"].strip()
            target = msgs[2]["content"] if len(msgs) > 2 else ""
            score = 1.0 if gen_text == target.strip() else 0.0
            depth_scores[depth].append(score)
    else:
        import torch
        for i in range(0, len(eval_rows), B):
            batch_rows = eval_rows[i:i + B]
            for r in batch_rows:
                depth = r.get("provenance", {}).get("source_depth", -1)
                msgs = r.get("messages", [])
                prompt_msgs = msgs[:2]
                prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                target = msgs[2]["content"] if len(msgs) > 2 else ""

                enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len)
                enc = {k: v.to(model.device if hasattr(model, "device") else device) for k, v in enc.items()}

                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)

                gen_text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                score = 1.0 if gen_text == target.strip() else 0.0
                depth_scores[depth].append(score)

    results = {}
    for d in sorted(depth_scores.keys()):
        scores = depth_scores[d]
        results[int(d)] = {"count": len(scores), "accuracy": sum(scores) / max(len(scores), 1)}

    in_dist_depths = [1, 2]
    transfer_depths = [3, 4]
    in_dist_scores = [s for d in in_dist_depths for s in depth_scores.get(d, [])]
    transfer_scores = [s for d in transfer_depths for s in depth_scores.get(d, [])]

    in_dist_acc = sum(in_dist_scores) / max(len(in_dist_scores), 1)
    transfer_acc = sum(transfer_scores) / max(len(transfer_scores), 1)
    crr = transfer_acc / max(in_dist_acc, 1e-8)

    return {
        "crr": crr,
        "in_distribution_accuracy": in_dist_acc,
        "transfer_accuracy": transfer_acc,
        "in_distribution_count": len(in_dist_scores),
        "transfer_count": len(transfer_scores),
        "by_depth": results,
    }


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent

    eval_path = args.eval_path or os.environ.get("BTL_EVAL_PATH", str(project_root / "data" / "real-traces" / "eval.jsonl"))
    if not Path(eval_path).exists():
        print(f"Error: eval dataset not found at {eval_path}")
        sys.exit(1)

    eval_rows = load_jsonl(eval_path)
    if args.limit:
        eval_rows = eval_rows[:args.limit]

    print(f"Loaded {len(eval_rows)} eval rows from {eval_path}")

    results = {}

    if args.adapter or args.gguf or args.model:
        model, tokenizer, is_gguf = load_model(args.model, args.adapter, args.gguf, args.device)
        r = evaluate_model(model, tokenizer, eval_rows, is_gguf, args.max_len, args.max_new_tokens, args.device)
        label = "3-loss model"
        if args.adapter:
            label = f"adapter: {args.adapter}"
        if args.gguf:
            label = f"gguf: {args.gguf}"
        print(f"\n=== {label} ===")
        print(f"  CRR:               {r['crr']:.4f}")
        print(f"  In-distribution:   {r['in_distribution_accuracy']:.4f} ({r['in_distribution_count']} rows)")
        print(f"  Transfer:          {r['transfer_accuracy']:.4f} ({r['transfer_count']} rows)")
        print(f"  By depth:          {json.dumps(r['by_depth'])}")
        results[label] = r

    if args.baseline_adapter:
        model_b, tokenizer_b, is_gguf_b = load_model(args.model, args.baseline_adapter, None, args.device)
        r_b = evaluate_model(model_b, tokenizer_b, eval_rows, is_gguf_b, args.max_len, args.max_new_tokens, args.device)
        label_b = f"SFT baseline: {args.baseline_adapter}"
        print(f"\n=== {label_b} ===")
        print(f"  CRR:               {r_b['crr']:.4f}")
        print(f"  In-distribution:   {r_b['in_distribution_accuracy']:.4f} ({r_b['in_distribution_count']} rows)")
        print(f"  Transfer:          {r_b['transfer_accuracy']:.4f} ({r_b['transfer_count']} rows)")
        print(f"  By depth:          {json.dumps(r_b['by_depth'])}")
        results[label_b] = r_b

    if args.adapter and args.baseline_adapter:
        diff = r['crr'] - r_b['crr']
        print(f"\n=== CRR Delta (3-loss - baseline) ===")
        print(f"  {r['crr']:.4f} - {r_b['crr']:.4f} = {diff:+.4f}")
        if diff > 0.05:
            print("  ✓ 3-loss pipeline improves compression")
        elif diff > 0:
            print("  ~ Marginal improvement")
        else:
            print("  ✗ Baseline performs better")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
