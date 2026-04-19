# Output Embedding Recombination

This document describes the current special heredity rules for the **trainable output-embedding tail rows**.

## Scope

This applies only to the **38 trainable values** associated with each action word's output-embedding vector.

It does not change:

- the shared input encoder
- the dense trunk
- the fixed 26-dimensional word-feature prefix of the output embedding

## Row-Level Recombination Policy

Each action word's 38-value trainable tail is treated as one recombination unit.

For one child row:

- with `99%` probability, copy the whole row from one parent
- with `1%` probability, use arithmetic recombination between the two parent rows

The arithmetic case samples one blend coefficient `lambda` for the whole row:

- use an edge-favouring beta-like sample
- map it into `[0.2, 0.8]`
- build the row as `lambda * parent_a + (1 - lambda) * parent_b`

So the common case is whole-row inheritance, but there is still a narrow recombinational path to genuinely mixed rows.

## Mutation Policy For Output Tail Rows

Output tails keep the same per-value mutation rate as the rest of the genotype, but they also have a row-level
mutation shape of their own.

### Per-Value Mutation

Per-value mutation is still small additive Gaussian drift.

### Whole-Row Magnitude Mutation

Each row also has a separate `0.5%` chance of whole-row scaling:

- `1.02`
- `0.98`

So the current behaviour is:

- mostly whole-row inheritance
- rare arithmetic row blending
- ordinary per-value drift
- rare whole-row magnitude adjustment
