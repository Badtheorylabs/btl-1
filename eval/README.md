# BTL-1 Eval

This directory holds the benchmark harness for BTL-1.

## Included scorers

- `btl1`: scores the project’s own held-out eval set
- `bfcl`: function-calling style tool selection and parameter matching
- `toolbench`: multi-step tool chains and dependencies
- `gaia`: freeform answer quality for multi-step reasoning tasks
- `crossos`: platform-agnostic OS tasks with portable shell intent scoring

## How the harness stays flexible

Each benchmark has a small manifest in `eval/specs/`. The manifests define:

- which field names or message paths to use for prompts
- which field names or message paths to use for expected answers
- which dataset path env var to honor
- which local file format(s) to accept

That keeps the harness from baking in one-off schema guesses all over the code.

## Optional benchmark files

Drop JSON, JSONL, parquet, or a directory of benchmark shards into `eval/data/` or point the runner at custom files with:

- `BTL_BTL1_PATH`
- `BTL_BFCL_PATH`
- `BTL_TOOLBENCH_PATH`
- `BTL_GAIA_PATH`
- `BTL_CROSSOS_PATH`

The notebook always runs the built-in BTL-1 held-out eval and any bundled benchmark files. External benchmarks are skipped when files are missing. The suite also reports contamination checks against the train split and each mounted benchmark file.
