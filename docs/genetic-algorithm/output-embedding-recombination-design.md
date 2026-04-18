# Output Embedding Recombination Design

This document records the intended future recombination and mutation policy for the **trainable output-embedding tail
rows**.

It is a design note, not a statement of current implementation. The current runtime still uses the same
per-parameter inheritance scheme across the whole genotype. This document describes the intended divergence for the
output embedding specifically.

## Scope

This design applies only to the **38 trainable values** associated with each action word's output-embedding vector.

It does **not** currently propose a matching change for:

- the shared input encoder
- the dense trunk
- the fixed 26-dimensional word-feature prefix of the output embedding

The reasoning is that an output-embedding tail row is already a meaningful semantic unit: it belongs to one specific
action word.

## Design Goal

The aim is to preserve a strong notion of whole-vector inheritance for output words, while still leaving a narrow path
for genuinely new vectors to arise through recombination and mutation.

In particular, the design tries to avoid two bad extremes:

- treating the output tails exactly like an unstructured flat float buffer
- copying whole rows so rigidly that novelty can arise only through very slow coordinate-wise drift

## Row-Level Recombination Policy

Each action word's 38-value trainable tail is treated as one recombination unit.

For a given child row:

- with **99% probability**, inherit the entire row from a single parent
- with **1% probability**, use arithmetic recombination between the two parent rows

The default and expected case is therefore whole-row single-parent inheritance.

Arithmetic recombination here means that the child row is formed by blending the two parent rows coordinate-wise rather
than by taking the entire row from one side unchanged. The exact blend formula can be fixed during implementation, but
the intended family of operators is simple arithmetic mixing rather than crossover-point slicing.

## Why Not Crossover Points Within a Row

The 38 trainable tail values are continuous latent coordinates, not an ordered chromosome with obvious contiguous
building blocks.

Because of that, a crossover point inside a row would impose an arbitrary left/right structure on a vector whose index
order is mainly representational. Arithmetic recombination is a better fit for this kind of parameter block.

## Mutation Policy For Output Tail Rows

For the time being, the mutation **rate** should stay aligned with the rest of the genotype.

That means this design does **not** currently propose a special higher mutation probability for output tails.

However, it does propose output-tail-specific mutation **shapes**.

### Per-Value Mutation

Per-value mutation should remain small additive drift:

- increment or decrement an individual value by a modest amount

This keeps the tails capable of local adaptation without making them too unstable.

### Whole-Row Magnitude Mutation

In addition to per-value mutation, each output-tail row should have a separate row-level mutation opportunity.

For each row:

- give this mutation its own separate probability, rather than tying it to the per-value mutation rate
- start with a low default probability of **0.5%** for a given child row
- when it triggers, scale the row magnitude by either **+2%** or **-2%**

In effect, the whole 38-value row is multiplied by either:

- `1.02`
- `0.98`

This is intended to give the GA a cheap way to strengthen or weaken an action word's learned trainable signature
without requiring many separate coordinate-level mutations to line up, while still keeping the effect small and rare.

## Intended Behavioural Effect

This design should make output-tail inheritance behave as follows:

- most child rows remain recognisably descended from one parent row
- a small minority of rows are genuine arithmetic mixtures of both parents
- local coordinate mutations can still fine-tune rows
- occasional row-level rescaling can produce faster shifts in a word's overall trainable influence

So the output embedding should retain stable per-word heredity while still allowing the population to discover new
useful action vectors over time.
