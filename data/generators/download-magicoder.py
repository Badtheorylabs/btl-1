import json, os, sys
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real-traces')
os.makedirs(OUT_DIR, exist_ok=True)

MAX_ROWS = int(os.environ.get('BTL_MAGICODER_MAX', '5000'))
SPLIT_EVAL = int(os.environ.get('BTL_MAGICODER_EVAL_PCT', '10'))

print(f'Downloading Magicoder-OSS-Instruct-75K (subset={MAX_ROWS})...')
ds = load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', streaming=True)

train_rows = []
eval_rows = []

for i, row in enumerate(ds):
    if i >= MAX_ROWS:
        break

    instruction = row['problem'].strip()
    solution = row['solution'].strip()

    if not instruction or not solution or len(solution) < 10:
        continue

    entry = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful coding assistant.'},
            {'role': 'user', 'content': instruction},
            {'role': 'assistant', 'content': solution},
        ],
        'provenance': {
            'template_id': f'magicoder-{i}',
            'source_depth': 0,
            'source_family': 'magicoder',
            'variant': 'verbose',
            'api_or_template': 'magicoder',
            'negative_type': None,
        },
    }

    if hash(f'magicoder-{i}') % 100 < SPLIT_EVAL:
        eval_rows.append(entry)
    else:
        train_rows.append(entry)

    if (i + 1) % 1000 == 0:
        print(f'  {i + 1}/{MAX_ROWS} processed')

train_path = os.path.join(OUT_DIR, 'train.jsonl')
eval_path = os.path.join(OUT_DIR, 'eval.jsonl')

with open(train_path, 'a') as f:
    for row in train_rows:
        f.write(json.dumps(row) + '\n')

with open(eval_path, 'a') as f:
    for row in eval_rows:
        f.write(json.dumps(row) + '\n')

print(f'Done. {len(train_rows)} train + {len(eval_rows)} eval = {len(train_rows) + len(eval_rows)} rows')
print(f'  train: {train_path}')
print(f'  eval:  {eval_path}')
