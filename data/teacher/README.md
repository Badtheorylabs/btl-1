# Teacher Pipeline

This directory is the staging area for BTL-1 teacher traces.

## What it does

- Reads `data/depth-1/train.jsonl` and `data/depth-2/train.jsonl`
- Builds three request queues for each source task:
  - `minimal`
  - `verbose`
  - `negative`
- Writes a combined `manifest.jsonl` plus one JSONL file per variant

## Current contract

- `minimal` keeps the gold tool chain and asks for the shortest correct reasoning wrapper.
- `verbose` keeps the gold tool chain and asks for a longer, still correct reasoning wrapper.
- `negative` asks for one controlled structural mistake so the trace can be used as a hard negative.

## Files written

- `manifest.jsonl`
- `minimal.jsonl`
- `verbose.jsonl`
- `negative.jsonl`

## Runner

The executable teacher runner lives at `data/teacher/run.mjs`.

It posts each job to an OpenAI-compatible chat endpoint and writes a run folder containing:

- `raw.jsonl`
- `completed.jsonl`
- `failed.jsonl`
- `summary.json`

## Usage

```bash
node data/teacher/pipeline.mjs
node data/teacher/run.mjs --dry-run --limit=5
```

Optional flags:

- `--depths=1,2` to limit which source depths are used
- `--limit=100` to cap rows per depth during smoke tests

Runner flags:

- `--base-url=https://.../v1`
- `--api-key=...`
- `--model=...`
- `--variants=minimal,verbose,negative`
- `--depths=1,2`
- `--limit=50`
- `--delay-ms=250`
- `--max-tokens=1024`
- `--temperature=0.2`
- `--dry-run`
