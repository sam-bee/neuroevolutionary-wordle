# Genetic Algorithm Design Overview

This document describes the current GA used by `run_genetic_algorithm`.

It focuses on runtime behaviour rather than file layout or low-level implementation detail. For storage and garbage
collection, see [`genotype-slab-design.md`](genotype-slab-design.md).

## Quick Answers

- **Is the GA generational or steady-state?**
  Generational.
- **Do we use elitism?**
  No.
- **How is parent selection done?**
  Spatially. Both parents are chosen by local rank-weighted roulette selection from the focal cell's configurable Moore
  neighborhood.
- **Do we use self-parenting?**
  No. The current cellular runtime requires two different parents.
- **What is fitness based on?**
  Simulated Wordle play.
- **Does population size stay fixed forever?**
  No. It is constrained by byte budgets and may shrink when genotype width grows.

## Overall Character

This is a population-based neuroevolution system for a Wordle-playing policy model.

Each organism carries:

- policy-model parameters
- trainable output-embedding tails for the currently active action space

The runtime proceeds in full generation steps:

1. evaluate the current population
2. build a complete child plan on the grid
3. breed and mutate a full child generation
4. replace the parent generation with the children

Parents and children may briefly coexist for memory-management reasons, but algorithmically this is still a
generational GA.

## Spatial Population Structure

The startup population is laid out on a square toroidal grid. Later generations may become rectangular if genotype
growth forces the runtime to reduce population size.

That means:

- one organism per cell
- top wraps to bottom
- left wraps to right
- startup population size is floored to a square number
- later population reductions remove whole rows from the maximum-y side while preserving the original column count

This same grid is also used by the spatial training-data shards.

## Parent Selection

The current runtime is a cellular GA.

For each child cell:

- both parents are chosen from the configured Moore neighborhood around that cell
- local candidates are ranked by normalized fitness, from worst rank `1` to best rank `N`
- equal-fitness candidates are randomly tie-broken before ranks are assigned
- each parent is sampled by roulette-wheel selection over `rank ^ parent_selection_rank_exponent`
- the two parents must be different
- the focal current-generation cell is eligible to be selected as one parent

The default breeding radius is `2`, which gives a `5 x 5` Moore neighborhood including the focal cell. The
`run_genetic_algorithm` CLI exposes this as `--breeding-radius N`. Radius uses the grid's toroidal Chebyshev/Moore
interpretation, so radius `1` is `3 x 3`, radius `2` is `5 x 5`, and edges wrap.

The default parent-selection rank exponent is `0.5`, exposed as `--parent-selection-rank-exponent F`.

When the next generation is smaller, no child is produced for cells in removed rows. Parents in those removed rows are
still eligible through the current generation's toroidal neighborhood during assembly-plan construction.

So the runtime is spatially local in mating, but still synchronous and generational in replacement.

## No Elitism

There is no elite carry-over.

Every organism in the next generation is produced as a child. If a strong lineage persists, it does so through
selection and reproduction, not because it is protected from replacement.

## Recombination and Mutation

The GA uses sexual recombination plus mutation.

For the encoder and dense trunk:

- inheritance is still fine-grained per-parameter mixing
- mutation is Gaussian drift

The output-embedding tails have their own special policy:

- the active tail table is usually copied from one parent-side source
- a level-2 crossover event can copy one randomly selected 38-value tail row from the alternate parent
- a level-3 crossover event can splice one randomly selected tail row at a feature crossover point
- rows still get ordinary per-value mutation
- whole-row `+2%` / `-2%` scaling exists in code, but the current CLI configuration leaves it disabled

For the exact current rules, see [`output-embedding-recombination.md`](output-embedding-recombination.md).

## Fitness Evaluation

Fitness is based on simulated Wordle play.

Each genome is scored by actually playing Wordle episodes, not by optimizing a supervised loss or differentiable RL
objective.

The current evaluator:

- builds a local training-word union for that organism's grid cell
- runs one fresh-grid and two prefilled-grid episodes per local training word
- rewards wins more when they happen earlier
- normalizes the final score into a positive `0..1`-like range for selection

For the exact current scoring scheme, see [`fitness-evaluation.md`](fitness-evaluation.md).

## Training and Action Space

The GA still grows from the top of one randomized action catalog, but new shards are released adaptively.

At any generation:

- the globally introduced training-word count
- and the globally selectable action-space count

are kept equal.

The important current distinction is:

- action-space growth is global
- fitness exposure is spatial

The initial foundation words are global from generation 0. Later introduced words become local training-data shards
that diffuse outward across the grid over time.

The first later shard is eligible no earlier than generation 10. After that, a later shard is eligible only after at
least 10 generations since the previous release, and then releases when either p99 fitness reaches the lower of a 0.05
gain over the previous release baseline or 0.70 total, or centroid distance mean has dipped below the configured
threshold, default 4.

## Genotype Growth and Population Size

This GA allows genotype size to grow over time as more action words become active.

When the action space grows:

- the output embedding needs more trainable tail rows
- the genotype becomes wider
- the runtime may need to reduce population size to stay within its configured memory budget

So population size is constrained by byte budgets, not only by a nominal organism count.

The practical consequence is:

- when genotype width stays the same, population size stays the same
- when genotype width grows, the allowed population may shrink
- when population shrinks, it shrinks by deleting complete rows rather than rescaling the whole grid

Local training-data shards keep stable two-dimensional epicentres. If a row deletion removes a shard epicentre's cell,
the epicentre's row is clamped upward to the last surviving row while its column is preserved.

## What This GA Is Not

To avoid ambiguity, the current runtime is **not**:

- an elitist GA
- a steady-state replacement system
- a mutation-only hill-climber
- a supervised training loop pretending to be a GA

It is a generational, spatially structured, recombination-plus-mutation neuroevolution system using episodic Wordle
performance as fitness.
