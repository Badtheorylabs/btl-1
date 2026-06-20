# BTL Compression Playbook

**Compression is the objective. Every other choice follows from that.**

This document is the operational guide to building BTL-1. It translates the compression thesis into concrete phases, task families, loss designs, and validation protocols. Each section is written to be implementable — no hand-waving, no "then a miracle occurs."

---

## Why Software Is the Test Domain

Coding is not an accidental choice. Software is one of the few domains where causal structure is unambiguous and measurable:

| Software concept | Causal analog |
|---|---|
| Code files | System state |
| Edits | Interventions |
| Tests | Causal feedback (did the intervention produce the expected effect?) |
| Diffs | Compressed action traces |
| Bug | System in undesired state |
| Fix | Intervention that transitions system to desired state |
| Repo repair | Planning under uncertainty — find the right intervention sequence |
| Wrong patch | Intervention that passes shallow checks but fails causal test |

A model trained on flat next-token prediction learns to complete code patterns. A model trained on compression should learn the **causal grammar of software** — what changes cause what effects, which dependencies are real, and how to trace symptoms to root causes.

**That is what separates "a model that writes code" from "a model that understands software."**

---

## Core Thesis (Sharpened)

Compression depth is not the number of steps in a chain. Compression depth is the **number of latent abstraction layers a model must learn to solve a task, such that the solution survives surface variation.**

- Depth 1: learn a lookup table (surface mapping, no abstraction)
- Depth 2: learn a rule that generates the mapping (abstracts over one degree of freedom)
- Depth 3: learn a meta-rule that generates the rule (abstracts over two degrees)
- Depth N: learn the causal structure that generates all observed variation

**What we test**: does a model trained on depth-N tasks transfer to unseen depth-N tasks with different surface features, better than it transfers to depth-(N+1) tasks? If yes, compression is working. If no, the model memorized.

**What we really want**: at depth 3+, the model should learn that code edits are causal interventions. A patch is not a string replacement — it is an action that changes system behavior in a predictable direction. That understanding is what compression, done right, should produce.

---

## Phase 1 — Curriculum by Abstraction Depth

Not a ladder of "harder" tasks. A ladder of **task families with controlled shifts**. Each family has a fixed latent structure and a manipulable surface layer.

### Task Families

#### Depth 1 — Single Tool Call
- **Structure**: map utterance → tool + params
- **Shift**: change tool name, param names, utterance phrasing
- **Transfer test**: novel utterance on same tool
- **Fail mode**: model memorizes utterance→tool pairs instead of learning tool schema

#### Depth 2 — Multi-Step Tool Chain
- **Structure**: plan sequence of dependent tool calls
- **Shift**: change which tools, change chain length (2 or 3), change tool order logic
- **Transfer test**: novel chain composition from seen tools
- **Fail mode**: model memorizes chain patterns instead of learning dependency resolution

#### Depth 3 — Code Edit Loop
- **Structure**: read code → identify bug / missing feature → edit → verify output
- **Causal frame**: an edit is an intervention on system state. A test passing is causal feedback. A wrong edit that compiles but fails the real test is an intervention that missed the causal target.
- **Shift**: change language (Python ↔ JS ↔ TS), change variable names, change indentation style
- **Transfer test**: same bug class in a different language or codebase
- **Fail mode**: model memorizes line-level edits instead of learning the repair structure
- **Why it matters**: this is the first depth where compression must encode causal structure, not just sequencing rules. If compression works, depth-3 is where the gap between compression-trained and SFT-trained models should be largest.

#### Depth 4 — Repo Repair
- **Structure**: navigate multi-file project → diagnose failure → make cross-file edits → run validation
- **Shift**: change project domain (web app ↔ CLI tool ↔ library), rename files, restructure directories
- **Transfer test**: novel repo with known failure pattern
- **Fail mode**: model memorizes file paths instead of learning project navigation

#### Depth 5 — Planning Under Uncertainty
- **Structure**: ambiguous task → explore → gather information → formulate plan → execute
- **Shift**: change domain entirely, change info-gathering strategy, change how ambiguity is resolved
- **Transfer test**: novel domain with same ambiguity class
- **Fail mode**: model always guesses instead of exploring

### Advancement Rule

Train depth N to convergence on all three losses. Then evaluate depth N under transfer shift:

> Minimum 80% on the depth-appropriate metric (see below) on held-out depth-N tasks where all surface features (names, paths, languages, phrasing) are replaced.

Only advance to depth N+1 when depth N is **stable** — the score under shift is within 10 points of the in-distribution score and above the absolute floor. Depth N+1 is a **probe**: probe it to measure zero-shot transfer, but do not use probe score as a promotion gate. Depth N should be solid before any depth N+1 training begins.

### Depth-Specific Metrics

`tool_orchestration_score` is sufficient for depth 1 (single tool) and depth 2 (chains). Depth 3+ require task-specific metrics because a model can produce valid JSON but fail at the actual task:

| Depth | Task | Primary Metric | Secondary Metrics |
|---|---|---|---|
| 1 | Single tool call | tool_orchestration_score | param accuracy, refusal rate |
| 2 | Multi-step chain | tool_orchestration_score | dependency accuracy, step count minimality |
| 3 | Code edit loop | **edit success rate** (edits that compile + pass existing tests) | test-pass rate, patch correctness (diff against ground truth), diff minimality (fewer changed lines = better compression) |
| 4 | Repo repair | **repair success rate** (all tests pass after patch) | cross-file edit correctness, hallucinated file rate, build time |
| 5 | Planning | **task completion rate** (user-goal met without over-help) | exploration efficiency (info-gathering steps before committing), recovery rate from wrong path |

All depth-specific scores are reported alongside `tool_orchestration_score` for depth 3+ tasks. The advancement rule uses the primary metric for the relevant depth.

This is not eval on a test split. This is eval on a **different distribution** that shares only the latent structure.

---

## Phase 2 — Teacher Traces as Compression Targets

The teacher (DeepSeek / GPT-4o / Claude) generates the training data, but with a specific protocol designed to support compression training, not just correctness. The first implementation lives in `data/teacher/pipeline.mjs`, which turns the depth-1 and depth-2 source sets into minimal, verbose, and negative request queues.

### Trace Generation Protocol

For each task in each depth family:

1. **Minimal correct trace** — prompt the teacher for the shortest correct solution. "Use as few steps as possible. Every step must be necessary."
2. **Verbose correct trace** — prompt the teacher for a correct solution with deliberately redundant steps or unnecessary param detail.
3. **Surface variants** — generate the same task with 3 different surface forms (different names, paths, phrasing, languages).
4. **Near-miss negative trace** — generate a trace that uses the wrong tool or wrong dependency order but produces the same final output by coincidence. This is the critical class.

### Dataset Structure

```
data/
  depth-1/
    tool_call/
      minimal.jsonl      # 500–1000 per tool
      verbose.jsonl      # 500 per tool
      negative.jsonl     # 200 per tool (near-miss)
      shifts/
        shift-A.jsonl    # transfer eval set
        shift-B.jsonl
  depth-2/
    chains/
      minimal.jsonl
      verbose.jsonl
      negative.jsonl
      shifts/
  depth-3/
    code_edit/
      ...
  depth-4/
    repo_repair/
      ...
  depth-5/
    planning/
      ...
```

Total target: ~5k–10k per depth level (50k total). Not 200k. Quality of compression signal >> quantity of surface examples.

---

## Phase 3 — Compression-Aware Training

Three losses, applied in order per depth level. The training for each depth level should converge before advancing.

### Loss 1: Correctness (NLL)

Standard supervised loss on minimal correct traces only. Establishes format, schema, and basic sequencing.

- **Data**: minimal.jsonl
- **Objective**: model learns to produce valid JSON tool chains
- **Stopping criterion**: eval accuracy > 90% on in-distribution held-out

### Loss 2: Minimality Preference

Preference training (DPO/ORPO/SimPO) that teaches the model that shorter correct traces are better.

- **Data pairs**: (minimal, verbose) — same task, two correct completions
- **Objective**: reward minimal traces, penalize verbose but correct ones
- **Signal**: the model should prefer the shortest trace that still works
- **Stopping criterion**: preference accuracy > 85% on unseen pairs

### Loss 3: Structural Contrastive

Contrastive training that teaches the model to distinguish "correct output" from "correct structure."

- **Data**: (minimal, near-miss negative) — same task, one structurally correct, one structurally wrong but output-matched
- **Objective**: the model learns to reject traces that get the right answer for the wrong reason
- **This is the core compression loss**: it forces the model to encode causal task structure, not output correlations
- **Stopping criterion**: rejection accuracy > 90% on unseen negatives

### Training Schedule

For each depth level N:
1. Train Loss 1 on depth-N minimal data (convergence)
2. Train Loss 2 on depth-N preference pairs (convergence)
3. Train Loss 3 on depth-N contrastive pairs (convergence)
4. Evaluate depth-N CRR: in-distribution score + transfer-under-shift score + absolute floor check
5. Probe depth-(N+1) zero-shot transfer — record but do not gate on it
6. Only advance to depth N+1 when depth N's CRR > 0.85 **and** raw transfer score is above the depth-specific floor

---

## Phase 4 — Validation Under Shift

The only eval that matters: **does behavior survive controlled surface variation?**

### Shift Types

| Shift | What changes | What stays same |
|---|---|---|
| Lexical | Variable names, file names, tool names in prompts | Task structure, tool schema |
| Formatting | Indentation, line spacing, comments style | Code logic, task intent |
| Language | Python ↔ JS ↔ TS ↔ pseudocode | Algorithm, repair pattern |
| Domain | Web app ↔ CLI tool ↔ data pipeline | Project structure, task type |
| Idiosyncrasy | Naming conventions, error message style | Failure pattern, recovery flow |

### Reporting

For every benchmark result, report two numbers:

1. **In-distribution score** — same surface as training
2. **Transfer-under-shift score** — surface replaced, structure preserved

If score(transfer) ≈ score(in-distribution), compression worked.
If score(transfer) ≪ score(in-distribution), compression failed — model memorized surface.

### Why CRR Is Necessary But Not Enough

CRR tells us whether behavior survives controlled surface shift, but it does not by itself prove causal compression. A model can be robust across lexical, formatting, or domain variants and still be doing very good memorization.

To close that gap, add a counterfactual probe:

- Keep the surface task valid.
- Violate the latent causal structure on purpose.
- Check whether the model rejects the trace or flags the inconsistency.

Examples:

- A code edit that compiles but changes the wrong function.
- A tool chain that reaches the same output through the wrong dependency order.
- A repo repair that passes one test while breaking the hidden cause of the failure.

Interpretation:

- High CRR plus good counterfactual rejection is evidence for causal compression.
- High CRR without counterfactual rejection is robust memorization.
- Low CRR is surface fragility.

---

## Expected Results

If the compression thesis is working, BTL should produce four concrete outcomes:

1. **Research result** - evidence that compression can be a training objective, not just a side effect of next-token prediction.
2. **Product result** - a small local coding agent that can handle file edits, shell commands, browser use, and repo repair on consumer hardware.
3. **Strategic result** - a model that wins by being smaller, sharper, and more transferable, not by chasing scale.
4. **Validation result** - high CRR under surface shift, with raw scores still above useful thresholds.

Success should look like a model that keeps its behavior when names, paths, phrasing, and code surface details change. It should remain fast enough to run comfortably on a 16 GB laptop, while still being useful for real coding and tool-use workflows.

### Primary Metric

**Compression Retention Ratio** = `transfer_score / in_distribution_score`

CRR is always reported alongside raw transfer and in-distribution scores. A high CRR with low raw scores is not evidence of compression — it is evidence of weak training.

**Requirement**: CRR > 0.85 **and** raw transfer score above the absolute floor for the task family:

| Depth | Minimum Transfer Score |
|---|---|
| 1 (single tool) | tool_orchestration_score > 75 |
| 2 (chains) | tool_orchestration_score > 70 |
| 3 (code edit) | edit success rate > 60% |
| 4 (repo repair) | repair success rate > 50% |
| 5 (planning) | task completion rate > 40% |

- CRR > 0.9 with raw scores above floor: strong compression
- CRR 0.7–0.9 with raw scores above floor: moderate compression
- CRR < 0.7 or raw scores below floor: surface memorization dominated or training insufficient

---

## Falsifiable Claim (Updated)

> A model trained on ~50k compression-curated examples across 5 depth levels (with preference + contrastive loss) will achieve a Compression Retention Ratio > 0.85 on depth-3 code-edit tasks, at 5x fewer total tokens seen than a standard SFT baseline trained on 200k flat examples.

The depth-3 code-edit level is the critical test because it is the first depth where **causal structure is unambiguous**: an edit is an intervention, a test is causal feedback, and a wrong edit that passes shallow checks but breaks the system is a causal error. If compression produces causal structure, depth-3 is where the gap between compression-trained and SFT-trained models should be largest.

**Null hypothesis**: standard SFT matches CRR at any token budget on any depth. If true, compression training is unnecessary and the thesis fails.

**To falsify**: run both pipelines (standard SFT vs. compression curriculum), measure CRR at matched compute on depth-1, depth-2, and depth-3. If CRR is indistinguishable at depth-3, the program needs a fundamentally different approach. If only depth-1 and depth-2 show a gap, the thesis holds for tool use but not for causal coding — a partial result worth publishing.

---

## Implementation Order

```
=== Week 1-2: proven pipeline (done) ===

Depth-1 specs + generators (10 tools, 2,000 rows)
Depth-2 specs + generators (13 chain patterns, 1,950 rows)
Shift protocol + all shift generators (13,800 total shift rows)
Coherent variable sampling across utterance and expected output

=== Week 3-4: generate + code-edit + baseline ===

Depth-3 code-edit spec + generator (minimal, verbose, negative traces)
Depth-4 repo-repair spec + generator (scaffolded, smaller set)
Teacher trace pipeline — fire real endpoint for depth 1-3
Post-process teacher output into train/dev splits
SFT baselines on depth-1, depth-2, depth-3 — measure CRR for each

=== Week 5-6: compression training ===

Compile depth-1: NLL + preference + contrastive training
Compile depth-2: same three losses
Compile depth-3: same three losses + counterfactual probe
Measure CRR for all three depths with absolute floor checks
Compare against SFT baselines — if CRR is indistinguishable, thesis fails early

=== Week 7+: depth 4-5 + publish ===

Phase 1:    Depth-5 (planning under uncertainty) task definitions
Phase 2:    Teacher trace generation for depth 4-5
Phase 3+4:  Train + validate each depth level against CRR thresholds
            Build counterfactual probe suite
            GGUF quantization + agent CLI validation
            Write results vs. null hypothesis
            Publish (weights, dataset, recipe, tech report)
```

---

## What We Are Not Doing

- Not chasing leaderboard benchmarks (BFCL, GAIA, SWE-bench, etc.) until CRR is validated
- Not training on 200k examples and hoping scale fixes structure
- Not using next-token prediction as the sole objective
- Not evaluating on in-distribution accuracy alone
- Not treating coding as an add-on task — it is the central causal domain
- Not claiming compression works until CRR > 0.85 with counterfactual rejection is reproducible
