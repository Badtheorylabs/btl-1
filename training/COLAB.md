# BTL-1 Colab Runbook

This follows the current Unsloth-based `train.py`.

## 1. Use a GPU runtime

In Colab, switch the runtime to GPU, then run cells top to bottom. Unsloth docs also mention the free T4 path.

## 2. Install dependencies

```python
!pip install -q unsloth datasets accelerate transformers sentencepiece
```

## 3. Put the repo on Colab

Clone your repo or mount Drive and copy it into `/content/btl-1`.

```python
!git clone <your-repo-url> /content/btl-1
%cd /content/btl-1
```

If your traces are on Drive, copy them into:

- `data/real-traces/train.jsonl`
- `data/real-traces/eval.jsonl`

`train.py` already defaults to those paths.

## 4. Run training

For a Colab-friendly run:

```python
import os

os.environ.update({
    "BTL_TRAIN_PATH": "/content/btl-1/data/real-traces/train.jsonl",
    "BTL_EVAL_PATH": "/content/btl-1/data/real-traces/eval.jsonl",
    "BTL_MAX_LEN": "768",
    "BTL_BATCH": "2",
    "BTL_GRAD_ACCUM": "8",
    "BTL_FLASH_ATTN": "0",
    "BTL_LORA_R": "32",
    "BTL_LORA_ALPHA": "64",
    "BTL_SFT_LIMIT": "12000",
    "BTL_DPO_LIMIT": "4000",
    "BTL_CONTRASTIVE_LIMIT": "1500",
    "BTL_DEEPSEEK_BUDGET_DOLLARS": "1.5",
    "BTL_OPENAI_API_KEY": "<your-openai-key>",
    "BTL_OPENAI_MODEL": "gpt-4o-mini",
})

!python train.py
```

If you want the generator stage to do the same split, run `data/generators/real-trace-generator.mjs` with the same DeepSeek key plus the OpenAI key above.

## 5. Download the adapter

```python
from google.colab import files
!zip -r /content/compression-adapter.zip /content/btl-1/artifacts/compression-adapter
files.download("/content/compression-adapter.zip")
```

## Notes

- `train.py` already uses `FastLanguageModel`.
- Unsloth's README says training can be up to 2x faster with lower VRAM use.
- If you want to run the notebook as a batch job, Colab docs support `Run all`.
