import json
import sys
from transformers import AutoTokenizer

path = r"C:\Users\pc\Downloads\btl\btl-1\data\final\train.jsonl"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

lengths = []
total = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        tokens = len(tokenizer.encode(text))
        lengths.append(tokens)
        total += 1
        if total % 50000 == 0:
            print(f"  processed {total}")

lengths.sort()
p50 = lengths[len(lengths) // 2]
p95 = lengths[int(len(lengths) * 0.95)]
p99 = lengths[int(len(lengths) * 0.99)]
mx = max(lengths)
avg = sum(lengths) / len(lengths)

print(f"\nRows checked: {len(lengths)}")
print(f"Avg:   {avg:.0f}")
print(f"P50:   {p50}")
print(f"P95:   {p95}")
print(f"P99:   {p99}")
print(f"Max:   {mx}")
