# Genetic Algorithm Design Overview

This document describes what kind of genetic algorithm this project currently uses.

It is intentionally about algorithmic behaviour and design choices rather than code structure, file layout, or
low-level implementation details. For the storage and garbage-collection side of the runtime, see the separate
genotype-slab design document.

## Quick Answers

- **Is the GA generational or steady-state?**
  Generational.
- **Do we use elitism?**
  No.
- **Do we use roulette-wheel selection?**
  No.
- **What parent selection do we use?**
  Tournament selection.
- **Can parents self-parent?**
  It is configurable in principle, but the current runtime defaults to no self-parenting.
- **What kind of thing is fitness based on?**
  Simulated Wordle play, not a supervised loss function.
- **Is this mainly asexual mutation-only evolution?**
  No. It is a sexual recombination-plus-mutation GA.
- **Does population size stay fixed forever?**
  Not necessarily. It is budget-constrained and may shrink when genotype size grows.

## Overall Character

This project uses a population-based neuroevolution approach for a Wordle-playing policy model.

Each organism carries:

- policy-model parameters
- trainable output-embedding tails for the currently active action space

The GA is not a steady-state system where individual organisms are continuously inserted and removed one at a time.
Instead, it works in generation steps:

1. evaluate the current population
2. select parent pairs from that evaluated population
3. breed and mutate a full child generation
4. replace the parent generation with the children

So although the runtime may briefly keep parents and children live at the same time for memory-management reasons, the
algorithm itself is generational.

## No Elitism

The current GA does **not** use elitism.

That means:

- there is no automatic carry-over of the best organism
- there is no elite fraction copied unchanged into the next generation
- every member of the next generation is produced as a child

If a strong lineage persists, it does so because those organisms keep being selected as parents, not because they are
protected from replacement.

## Parent Selection

Parent selection is tournament-based.

That means the GA:

- samples a small candidate subset from the evaluated population
- chooses the fittest individual from that subset as a parent
- repeats for the second parent

This is deliberately **not** roulette-wheel or fitness-proportionate selection. Selection pressure comes from
tournament winners rather than from assigning each organism a probability mass proportional to fitness.

It is also not currently a rank-based selector.

The current default behaviour is:

- tournament size of 3
- no self-parenting

Only evaluated organisms are eligible to be selected.

## Reproduction

The GA uses sexual reproduction with mutation.

Each child is produced from two selected parents by:

1. recombining parameters from the two parents
2. mutating the resulting child genome

The recombination style is per-parameter inheritance rather than copying large contiguous chromosome blocks. In effect,
each child genome is assembled from fine-grained choices between the two parents, and then mutation perturbs the result.

The mutation model is probabilistic Gaussian noise applied to parameters, not a bit-flip scheme.

At present that statement is true across the whole genotype. However, the intended design direction is to let
different genotype components use different recombination policies when that better matches their structure. In
particular, the trainable output-embedding tails are planned to move toward mostly row-level inheritance with rare
arithmetic recombination; see
[`output-embedding-recombination-design.md`](output-embedding-recombination-design.md).

## Fitness Evaluation

Fitness is based on simulated Wordle play.

This is important because it means the GA is not currently optimising:

- a supervised cross-entropy loss
- a differentiable reinforcement-learning objective
- a hand-labelled target policy

Instead, each genome is scored by actually using it to play Wordle episodes over the current active training-word
shard.

The current fitness signal is episodic and environment-based:

- each active training word contributes multiple evaluation episodes
- the runtime currently uses one fresh-grid episode and two partially prefilled-grid episodes per training word
- wins score positively
- earlier wins score higher than later wins
- failures score zero

Overall fitness is the sum of those episode scores across the current shard.

So the fitness function is closer to an aggregate task-performance score than to a conventional machine-learning loss.

For the exact currently implemented scoring scheme, see
[`current-fitness-evaluation.md`](current-fitness-evaluation.md).

## Training and Action Space

The GA currently works against a phased prefix of the curated Wordle action catalog, but fitness exposure is spatial
rather than globally mixed.

At any generation there are two closely related global counts:

- the active training-word count
- the selectable action-space count

At present those are kept equal in the runtime.

By default the foundation prefix starts at 20 words and stays fixed unless a word-count growth schedule is configured.
When the schedule introduces more words, those new words are added to every organism's action space immediately, but
their fitness influence diffuses through the cellular grid as spatial training-data shards rather than hitting the whole
population at once.

## Genotype Growth

This GA allows genotype size to grow over time as the active action space grows.

When more action words become active:

- the output embedding needs more trainable tail rows
- the genotype becomes wider
- the runtime may need to reduce population size to stay within its configured memory budget

So this is not a GA where genome size is assumed fixed for the entire run.

## Population Size Policy

Population size is constrained by byte budgets, not only by a nominal organism count.

The important practical consequence is:

- when genotype width stays the same, population size stays the same
- when genotype width grows, the allowed population may shrink

This is a deliberate part of the design rather than an accidental side effect.

## Evaluation and Replacement Timing

Children are assembled only after the current generation has been evaluated.

Newly assembled children start unevaluated. They become the evaluated parent population only on the next generation
step.

This means the GA has a clean alternating rhythm:

- evaluated parent generation
- unevaluated child generation
- evaluate
- breed next generation

## What This GA Is Not

To avoid ambiguity, the current design is **not**:

- an elitist GA
- a roulette-wheel GA
- a steady-state replacement system
- a mutation-only hill-climber
- a supervised training loop pretending to be a GA

It is a generational, tournament-selected, recombination-plus-mutation neuroevolution system using episodic Wordle
performance as fitness.
