# Training Data Sharding Curriculum

This document describes the **current** curriculum used to expose training words to the GA runtime.

It is a description of the implemented schedule and terminology baseline, not a final specification for the next
redesign.

## Current High-Level Model

The project currently loads a single randomized action-space catalog of `4,739` words.

At process start:

- the full catalog is loaded from `data/action-space-randomised.txt`
- the full catalog is uploaded to GPU constant memory once

During a run, the GA does **not** currently switch between independent hand-authored shards.

Instead, it uses a **monotonic active prefix** of that single randomized catalog.

So in current terms:

- the “training data shard” is effectively the first `N` words of the randomized catalog
- later phases expand that shard by increasing `N`
- words are never removed once they have become active

## Current Runtime Counts

The runtime currently keeps these counts equal:

- active training-word count
- active selectable action-space count

That means the same active prefix is used both for:

- choosing which solution words are evaluated
- choosing which action words the genome may output

## The Schedule

The implemented curriculum is defined by three numbers:

- `initial_word_count`
- `word_count_step`
- `word_count_step_period_generations`

The current default values are:

- `initial_word_count = 20`
- `word_count_step = 0`
- `word_count_step_period_generations = 1`

So by default the shard is fixed at the first `20` words for the whole run.

## Per-Generation Rule

For generation `g`, the active shard size is:

```text
active_word_count(g) =
    min(catalog_word_count,
        initial_word_count + floor(g / word_count_step_period_generations) * word_count_step)
```

with the special case that if `word_count_step = 0`, the active word count remains fixed at
`initial_word_count`.

So the implemented curriculum is:

- prefix-based
- deterministic
- monotonic
- count-driven rather than identity-driven

## What a Phase Means Right Now

In the current runtime, a **phase** is simply a contiguous run of generations with the same active shard size.

So if:

- `initial_word_count = 20`
- `word_count_step = 10`
- `word_count_step_period_generations = 5`

then the phases are:

- generations `0` to `4`: first `20` words
- generations `5` to `9`: first `30` words
- generations `10` to `14`: first `40` words
- generations `15` to `19`: first `50` words

and so on until the full catalog is active.

## What Happens When a Phase Grows

When the active shard grows:

- the training/evaluation task grows
- the selectable action space grows
- the genotype widens because more output-embedding tail rows are needed
- newly activated output rows are injected during next-generation assembly
- population size may shrink if the fixed memory budget can no longer sustain the previous population at the wider
  genotype size

So a phase transition is not only a data change. It is also a genotype-size and capacity change.

## What This Is Not

The current curriculum is **not** yet:

- a set of named disjoint shards
- a rotation over multiple shard identities of the same size
- a spatially aware shard layout
- a curriculum where training words and action words differ
- a validation/test split system

It is only an expanding active prefix over one randomized catalog.

## Why “Shard” Needs Clarifying

If the project starts talking about “training data shards”, that word currently maps only loosely onto the
implementation.

Today, the closest implemented meaning is:

- “the active prefix currently exposed to evaluation and action selection”

That is useful as a baseline, but it is much weaker than a proper sharding design.

## Practical Consequences of the Current Curriculum

The current approach has a few important properties:

- it is simple to implement
- it is deterministic for a given schedule
- it makes curriculum growth easy to reason about
- it couples training-task growth directly to output-embedding growth
- it can make generation-to-generation fitness harder to compare when the active count changes aggressively

Those consequences matter when designing the next version of sharding.

## Design Baseline for the Next Rewrite

For future discussion, the current baseline can be summarized as:

- one randomized master catalog
- one active prefix
- identical training-word and action-word exposure
- phase changes driven only by active-count growth
- no true independent shard identities yet

That is the system the next sharding design will be replacing or generalizing.
