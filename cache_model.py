from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("downloading tokenizer...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
print("tokenizer cached")
print("downloading model (this takes 1-2 min)...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("model cached")
del model
