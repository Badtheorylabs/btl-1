# BTL-1 Architecture

## What Makes This Groundbreaking

The field trains models to predict tokens. Compression emerges incidentally — shallow by default.

BTL-1 inverts this: **compression is the objective, not the side effect**. The entire system is built to test one falsifiable claim:

> A small local model trained on ~50k compression-curated examples will outperform a model trained on 200k flat examples at transfer under surface shift, at 5x fewer total tokens seen.

If true, it changes what training means — from "more data, more parameters" to **"better structure per token."**

---

## System Diagram

```
                          ┌──────────────────────────────────────┐
                          │         TEACHER (DeepSeek V4 Flash)   │
                          │  Generates 3 trace variants per task   │
                          └──────────┬───────────────┬───────────┘
                                     │               │
                    ┌────────────────┼───────────────┼──────────────────┐
                    │                │               │                  │
                    ▼                ▼               ▼                  ▼
             ┌──────────┐    ┌──────────┐    ┌──────────┐     ┌──────────────┐
             │ MINIMAL  │    │ VERBOSE  │    │ NEGATIVE │     │COUNTERFACTUAL│
             │ correct  │    │ correct  │    │ near-miss│     │  (depth 3+)  │
             │ shortest │    │ verbose  │    │  wrong   │     │  probe only  │
             └─────┬────┘    └─────┬────┘    └─────┬────┘     └──────────────┘
                   │               │               │
                   └───────┬───────┴───────┬───────┘
                           │               │
                           ▼               ▼
              ┌─────────────────────────────────────────────┐
              │         COMPRESSION TRAINING PIPELINE        │
              │                                              │
              │  ┌──────────────────────────────────────┐    │
              │  │   LOSS 1: NLL on minimal traces       │    │
              │  │   (learns format + correctness)       │    │
              │  └──────────────────┬───────────────────┘    │
              │                     │                         │
              │  ┌──────────────────▼───────────────────┐    │
              │  │   LOSS 2: PREFERENCE (DPO)           │    │
              │  │   minimal > verbose on same task      │    │
              │  │   (learns to prefer shorter traces)   │    │
              │  └──────────────────┬───────────────────┘    │
              │                     │                         │
              │  ┌──────────────────▼───────────────────┐    │
              │  │   LOSS 3: CONTRASTIVE                 │    │
              │  │   minimal > near-miss negative         │    │
              │  │   (learns to reject wrong structure)   │    │
              │  │   ★ CORE COMPRESSION SIGNAL ★         │    │
              │  └──────────────────┬───────────────────┘    │
              │                     │                         │
              └─────────────────────┼─────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │     QWEN 2.5 CODER 7B       │
                    │     + LoRA adapter           │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────────┐
              │           VALIDATION (CRR PROTOCOL)          │
              │                                              │
              │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
              │  │IN-DIST   │  │ TRANSFER │  │COUNTER-  │  │
              │  │SCORE     │  │ UNDER    │  │FACTUAL   │  │
              │  │(same     │  │ SHIFT    │  │PROBE     │  │
              │  │surface)  │  │(lexical, │  │(reject   │  │
              │  │          │  │format,   │  │wrong     │  │
              │  │          │  │domain,   │  │structure)│  │
              │  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
              │       │             │              │        │
              │       └─────────────┼──────────────┘        │
              │                     │                        │
              │            CRR = transfer / in-dist          │
              │       Requires CRR > 0.85 + raw floor        │
              └─────────────────────────────────────────────┘
```

---

## Depth Ladder

```
DEPTH 1                    DEPTH 2                   DEPTH 3
Single tool call      Multi-step chain          Code edit loop
┌──────────┐          ┌──────────────┐          ┌──────────────┐
│ "Find    │          │ "Search X,   │          │ "Fix off-by- │
│ budget   │──────→   │  open Y,     │────────→ │  one in      │
│ PDF"     │ file_    │  email Z"    │ chain    │  sum_to_n()" │
│          │ search   │              │          │              │
└──────────┘          └──────────────┘          └──────────────┘
                                                            │
                                                            │
                      DEPTH 5                     DEPTH 4    │
                 Planning under uncertainty      Repo repair │
                 ┌─────────────────────┐    ┌──────────────┐ │
                 │ "Something is wrong │    │ "The app     │ │
                 │  with the CI but    │←───│  crashes     │ │
                 │  I don't know what" │    │  when marking│ │
                 │                     │    │  task done"  │ │
                 └─────────────────────┘    └──────────────┘ │
                                                            │
                    ▲ Causal structure increases ────────────┘
                    │ Each depth adds another abstraction layer
                    │ that must survive surface variation
```

---

## Three-Loss Training Detail

```
Standard SFT:                    BTL-1 Compression Training:
──────────────                    ────────────────────────────

     Data                             Data
      │                                │
      ▼                                ├── Minimal ──► Loss 1 (NLL)
  [NLL Loss]                           │    (learns format)
      │                                ├── Verbose ──► Loss 2 (DPO)
      ▼                                │    (prefers minimal)
  "Predict next token"                 │
                                      └── Negative ─► Loss 3 (Contrastive)
                                           (rejects wrong structure)

                                              │
                                              ▼
                                     "Compress the causal program"
```

---

## What the Final Model Looks Like

**Name**: BTL-1 (Bad Theory Labs — Model 1)

**Base**: Qwen 2.5 Coder 7B + QLoRA adapter (rank 64, alpha 128)

**Size**: ~4.5 GB (Q4_K_M GGUF)

**Speed**: 20-30 tok/s on i7/16GB laptop

**Behavior**:

| Scenario | What it does | Why it's different |
|---|---|---|
| "Fix this bug" | Reads the file, identifies the causal failure, writes the minimal correct patch | Doesn't just complete code — it understands which edit is the right intervention |
| "Search for X and email the result" | Plans a 2-step chain: web_search → email, passes output between steps | Tool orchestration is compressed into dependency resolution |
| "The app crashes when I click submit" | Navigates the project, reads the error handler, traces the symptom to the root cause file | Repo repair as causal tracing, not file-pattern matching |
| Same task, different variable names | Produces the same fix with different identifiers | The fix is bound to the causal structure, not the surface tokens |
| Same bug, different language | Transfers the fix pattern from Python to JavaScript | Compression depth survived the language shift |

**What it cannot do** (by design):
- Chat about general topics
- Roleplay or creative writing
- Answer factual questions outside coding/tools
- Follow vague instructions without tool context

It is not a chatbot. It is a **compression-first coding agent** — narrow by design, deep where it matters.

---

## The Bet In One Diagram

```
Performance
under shift
    ▲
    │                    ★ BTL-1 (compression-trained)
    │                   /
    │                  /  ← CRR gap
    │                 /
    │                ──────── Standard SFT baseline
    │               /
    │              /
    │             /
    │────────────► Tokens seen during training
    │
    └───────────────────────────────────►
        5x fewer tokens    200k tokens
```

If the gap exists at 5x fewer tokens: **thesis confirmed, compression works.**
If no gap: **thesis falsified, compression training is just expensive SFT.**

Both results are publishable. That is the point.
