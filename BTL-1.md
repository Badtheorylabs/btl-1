# BTL-1 - Compression-First Local Agent

BTL-1 is a compression-first training program for a small local model that can do serious agentic coding on consumer hardware. The target is not a generic chat bot. The target is a compact 7B-class model that is strong at coding, tool use, browser/file automation, and repo repair because it learns the shortest causal program that explains the task.

Base model: `Qwen/Qwen2.5-Coder-7B-Instruct`.

Open weights. Open dataset. Published recipe.

---

## Core Thesis

> A small model can become a much better local agent if we train it to compress task structure, not just predict tokens.

The practical claim is simple:

- better transfer under surface shift
- fewer training tokens for the same or better behavior
- stronger coding and repo repair than a plain SFT baseline at the same model size

The compression idea is not a metaphor. It is the training objective.

---

## What We Are Optimizing

The real target for BTL-1 is:

- frontier-like coding for a 7B model
- strong autonomous repo repair
- broad reasoning that helps with tool use and planning

The method is:

1. Train on canonical prompts, not bloated variants.
2. Reward minimal correct traces.
3. Penalize verbose or wrong-structure traces.
4. Measure transfer under controlled shift, not just in-distribution accuracy.

That is the compression thesis in operational form.

---

## Training Plan

BTL-1 is moving from a stale SFT-only loop to a staged compression pipeline:

1. **NLL stage** - teach the shortest correct traces.
2. **Preference stage** - prefer minimal traces over verbose ones.
3. **Contrastive stage** - reject near-miss negatives that look plausible but break causal structure.

The default sequence length is intentionally tight. The repo is tuned for short, information-dense examples instead of giant token dumps.

---

## Current State

### Built

- Depth-1 and depth-2 task family scaffolds
- Depth-3 code-edit scaffolding
- Depth-4 repo-repair scaffolding
- Teacher manifest and runner
- Shift protocol and benchmark harness
- Canonical prompt structure for small, shift-resistant traces

### In Progress

- Real teacher completions for the training corpus
- The new compression-first training script
- CRR reporting in the benchmark runner

### Important Note

The current `data/final` corpus still contains dry-run placeholder outputs in some paths. That is fine for plumbing tests, but it is not the final supervision signal. The training pipeline should use real teacher completions and should fail fast if it only finds placeholders.

---

## Evaluation

The main metric is **Compression Retention Ratio**:

`CRR = transfer_score / in_distribution_score`

We care about both numbers:

- raw in-distribution quality
- transfer under surface shift

If the model only looks good in-distribution, the compression claim is weak.

---

## Falsifiable Claim

> A 7B model trained with compression-first objectives on curated teacher traces will match or beat a plain SFT 7B baseline on agentic coding transfer, while using fewer tokens and staying compact enough for local use.

If the model does not transfer better, the thesis fails.

---

## Repository Map

```
btl-1/
|-- BTL-1.md
|-- PLAYBOOK.md
|-- COMPRESSION.md
|-- train.py              # compression-first training entry point
|-- eval/                 # benchmark harness with CRR reporting
|-- data/
|   |-- specs/            # task family definitions
|   |-- generators/       # synthetic and template generators
|   |-- teacher/          # manifest, runner, and completed teacher runs
|   |-- depth-1/
|   |-- depth-2/
|   |-- depth-3/
|   |-- depth-4/
|   `-- final/
`-- artifacts/
```
