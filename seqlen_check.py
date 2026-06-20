import json
import sys
from transformers import AutoTokenizer

path = sys.argv[1] if len(sys.argv) > 1 else "/home/zeus/btl-1/data/final/train.jsonl"
n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
lengths = []
with open(path) as f:
    for idx, line in enumerate(f):
        if idx >= n_max:
            break
        row = json.loads(line)
        text = tk.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        lengths.append(len(tk.encode(text)))

lengths.sort()
n = len(lengths)
print(f"n={n}")
print(f"avg={sum(lengths)//n}")
print(f"p50={lengths[n//2]}")
print(f"p90={lengths[int(n*0.9)]}")
print(f"p95={lengths[int(n*0.95)]}")
print(f"p99={lengths[int(n*0.99)]}")
print(f"max={max(lengths)}")
