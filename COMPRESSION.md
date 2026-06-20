# BTL Compression Note

## Core Idea

Compression is the central thesis, not a side effect.

The program assumes that intelligence can be understood as compressing experience into representations that keep what matters and discard what does not. In this view:

- perception is compression of raw input
- reasoning is compression of representations into useful structure
- action is compression of a goal plus a world model into a decision

The important claim is not just that models become smaller. It is that a good compression process may produce representations that are more stable, more transferable, and more causally structured.

## What Compression Means Here

This project uses compression in a stronger sense than simple parameter count or model quantization.

Compression here means:

- learn the shortest useful representation of a task or distribution
- preserve the structure needed for prediction under change
- avoid memorizing surface patterns when a rule or causal model is available
- turn a large teacher signal into a smaller student that keeps the behavior we care about

So the goal is not just a tiny model. The goal is a tiny model that still behaves like a capable agent.

## Compression Depth

Compression depth is the number of abstraction steps needed to reduce a problem to its minimal sufficient description.

Examples:

- depth 1: memorize a mapping
- depth 2: learn the rule that generates the mapping
- depth 3: learn the meta-rule behind the rule

The hypothesis is that deeper compression should improve generalization, especially on tasks that require transfer, planning, or counterfactual reasoning.

## Causal Minimality

A compressed representation is only interesting if it keeps the generating structure of the data, not just a statistically compact summary.

That means the representation should support:

- interventional reasoning, not only observational pattern matching
- stable behavior when irrelevant details change
- correct transfer to new tasks that share the same underlying structure

If the model only compresses correlations, it may look good on familiar inputs but fail when the task shifts.

## How BTL Fits

BTL is the proof vehicle for this thesis.

Instead of trying to make one model do everything, BTL should test whether a smaller local model can be compressed into strong:

- coding behavior
- tool use
- browser and file automation
- stepwise planning

This is why the BTL direction is now:

- base a small shipped model on a code-capable open model
- use stronger teacher signals to generate high-quality traces
- compress those traces into a smaller student
- validate on real coding and agentic tasks

## What We Want To Prove

The concrete claim is:

- a smaller local model can preserve useful agentic coding behavior after compression
- the resulting model can still run comfortably on consumer hardware
- the compressed model can be better aligned to the product goal than a generic larger model

This is not a benchmark-chasing exercise. It is a test of whether compression can produce a better structure, not merely a smaller artifact.

## Validation Shape

The validation should answer three questions:

1. Does compression preserve coding skill?
2. Does compression preserve tool use and multi-step execution?
3. Does the compressed model stay fast and comfortable on a 16 GB laptop?

That means the eval should include:

- code editing and repair
- repo navigation
- tool-call correctness
- end-to-end task completion
- latency and RAM measurements

## Working Summary

BTL is trying to show that compression is not just model shrinking.

It is a method for building a smaller system that keeps the right structure:

- smaller, but still capable
- fast, but still deliberate
- local, but still useful
- specialized, but still general enough to operate as an agent

If this works, BTL is not only a model release. It is evidence for the compression thesis itself.
