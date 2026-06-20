"""
Convert trained QLoRA adapter + base model to GGUF format.

Usage:
  python scripts/convert-to-gguf.py --adapter artifacts/compression-adapter --output model.gguf
  python scripts/convert-to-gguf.py --adapter artifacts/compression-adapter --output model-q4_k_m.gguf --quantize q4_k_m

Requires:
  pip install torch transformers unsloth llama-cpp-python
  # or for conversion only: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert QLoRA adapter to GGUF")
    parser.add_argument("--adapter", required=True, help="Path to trained adapter directory")
    parser.add_argument("--base-model", default=os.environ.get("BTL_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
                        help="Base model name or path")
    parser.add_argument("--output", default="compression-model.gguf", help="Output GGUF file path")
    parser.add_argument("--quantize", default=None,
                        choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization type (default: f16, no quantization)")
    parser.add_argument("--llama-cpp-dir", default=None,
                        help="Path to llama.cpp directory (auto-cloned if not provided)")
    parser.add_argument("--keep-hf", action="store_true", help="Keep intermediate HF merged directory")
    parser.add_argument("--device", default="auto", help="Device for merging (auto/cpu/cuda)")
    return parser.parse_args()


def merge_adapter(adapter_path: Path, base_model: str, output_path: Path, device: str):
    print(f"Merging adapter {adapter_path} with base {base_model}...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device if device != "auto" else "auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("Loading and merging adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print("Merge complete.")
    return output_path


def convert_to_gguf(hf_path: Path, gguf_path: Path, llama_cpp_dir: Path | None, quantize: str | None):
    if llama_cpp_dir is None:
        llama_cpp_dir = Path(tempfile.mkdtemp()) / "llama.cpp"
        if not (llama_cpp_dir.parent / "llama.cpp" / "convert.py").exists():
            print("Cloning llama.cpp...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp_dir)],
                check=True, capture_output=True
            )
    else:
        llama_cpp_dir = Path(llama_cpp_dir)

    convert_py = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        print("Error: convert_hf_to_gguf.py not found in llama.cpp directory")
        sys.exit(1)

    print(f"Converting to GGUF: {hf_path} -> {gguf_path}")
    cmd = [
        sys.executable, str(convert_py),
        str(hf_path),
        "--outfile", str(gguf_path),
    ]
    if quantize:
        outdir = gguf_path.parent
        stem = gguf_path.stem
        if quantize == "q4_0":
            cmd.extend(["--outtype", "q4_0"])
        elif quantize == "q4_k_m":
            cmd.extend(["--outtype", "q4_k_m"])
        elif quantize == "q5_k_m":
            cmd.extend(["--outtype", "q5_k_m"])
        elif quantize == "q8_0":
            cmd.extend(["--outtype", "q8_0"])
        else:
            cmd.extend(["--outtype", "f16"])

    subprocess.run(cmd, check=True)
    print(f"GGUF model saved to {gguf_path}")


def main():
    args = parse_args()
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"Error: adapter directory {adapter_path} not found")
        sys.exit(1)

    gguf_path = Path(args.output)
    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    hf_merged = gguf_path.parent / "merged-hf"
    merge_adapter(adapter_path, args.base_model, hf_merged, args.device)

    convert_to_gguf(hf_merged, gguf_path, Path(args.llama_cpp_dir) if args.llama_cpp_dir else None, args.quantize)

    if not args.keep_hf:
        print("Cleaning up merged HF directory...")
        shutil.rmtree(hf_merged, ignore_errors=True)

    file_size = gguf_path.stat().st_size / (1024**3)
    print(f"Done. GGUF model: {gguf_path} ({file_size:.2f} GB)")


if __name__ == "__main__":
    main()
