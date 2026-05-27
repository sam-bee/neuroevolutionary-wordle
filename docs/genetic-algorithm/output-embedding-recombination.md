# Output Embedding Recombination

This document describes the current special heredity rules for the **trainable output-embedding tail rows**.

## Scope

This applies only to the **38 trainable values** associated with each active action word's output-embedding vector.

It does not change:

- the shared input encoder
- the dense trunk
- the fixed 26-dimensional word-feature prefix of the output embedding

## Current Row-Level Recombination Policy

The implemented device path does not use arithmetic row blending.

For the output-tail table, the child starts by copying every active row from one parent-side source. With the default
runner configuration:

- `crossover_temperature_level1 = 0.02` controls whether the whole output-tail table flips away from the base source
- `crossover_temperature_level2 = 0.01` can copy one randomly selected tail row from the alternate parent
- `crossover_temperature_level3 = 0.005` can splice one randomly selected tail row at a feature crossover point

The level-3 splice chooses a crossover point inside the 38-feature trainable tail. Features before that point come from
the row's current source and features after that point come from the alternate parent.

So the common case is still mostly whole-row inheritance, but the implemented rare mixed-row path is a one-point splice,
not `lambda * parent_a + (1 - lambda) * parent_b` arithmetic recombination.

## Mutation Policy For Output Tail Rows

Output tails keep the same per-value mutation rate as the rest of the genotype.

### Per-Value Mutation

Per-value mutation is small additive Gaussian drift. The current runner uses:

- `mutation_probability = 0.0001`
- `mutation_sigma = 0.02`

### Whole-Row Magnitude Mutation

The code still has support for whole-row magnitude scaling by `1.02` or `0.98`, controlled by
`output_tail_row_scale_mutation_probability`.

The current `run_genetic_algorithm` configuration leaves that probability at `0.0`, so row-scale mutation is disabled
in normal CLI runs unless a caller constructs a different assembly config in code.

So the current CLI behaviour is:

- mostly table/row copying from one parent-side source
- rare single-row alternate-parent copying
- rarer single-row one-point splicing
- ordinary per-value drift
- no whole-row magnitude adjustment by default
