# Training Data Sharding Curriculum

This document records:

- the previous active-prefix curriculum baseline
- the **current** implemented spatial training-data shard model

The old baseline still matters, because the implemented shard model keeps the same phased action-space growth schedule
while changing how those introduced words are exposed during fitness evaluation.

## Current Baseline

Today, the runtime uses one randomized action catalog of `4,739` words.

At process start:

- the full catalog is loaded from `data/action-space-randomised.txt`
- the full catalog is uploaded to GPU constant memory once

During a run, the old baseline did **not** switch between independent shard identities.

Instead, it uses a monotonic active prefix of that single randomized catalog. The relevant knobs are:

- `initial_word_count`
- `word_count_step`
- `word_count_step_period_generations`

In that baseline implementation:

- the active training-word count and active selectable action-space count are kept equal
- later curriculum phases are created only by increasing the active prefix length
- once a word becomes active, it stays active

So the current meaning of “shard” is only a loose one:

- the effective shard is just “the first `N` words of the randomized catalog”

That baseline matters because the current design deliberately changes the semantics of sharding without giving up the
existing idea that action-space growth widens the genotype.

## Current Spatial Shard Design

The runtime now moves away from “one expanding active prefix” and toward **spatially aware training-data shards**.

The core idea is:

- new words are introduced as shards
- each shard has a position on the same toroidal cellular grid used by the population
- each shard has an effective radius on that grid
- the shard affects fitness evaluation only for organisms whose cells lie within that radius

So the training-data curriculum is now a spatial diffusion process layered on top of the existing phased word-count
schedule.

## Relationship To The Cellular GA Grid

Training-data shards use the same conceptual grid system as the cellular GA.

That means:

- the population lives on a square toroidal grid
- each organism occupies one cell
- each training-data shard also has a cell position on that same toroidal grid
- shard coverage is determined by radius on that same toroidal grid

The spatial-cell concept is therefore shared runtime structure, not a property owned only by genotypes or only by
fitness evaluation.

## Shard Identity

Under the current design, a shard is a real object, not just a count.

Each shard has at least:

- a set of 5-letter words
- a center cell on the toroidal grid
- an effective radius
- a growth cadence for that radius

So unlike the current baseline, shards have identity, geometry, and temporal behavior.

## Action-Space Exposure

When a new training-data shard is introduced, its words are added to the action spaces of **all** organisms straight
away.

That means:

- output-embedding growth remains global
- genotype width still grows globally when new shard words are introduced
- slab pressure and population-capacity pressure are still global consequences of shard introduction

So this design is **not** trying to localize genotype width. It is localizing only the **fitness-evaluation influence**
of newly introduced words.

## Fitness-Evaluation Exposure

The local training data used for one organism’s fitness evaluation is the full union of all shards whose effective
radius covers that organism’s cell.

So for a given organism:

- find every shard currently in range of that organism’s cell
- take the full union of all words from those shards
- evaluate the organism against that full union

This means different organisms may be evaluated against different numbers of words at the same generation.

That is intentional.

## Weighting Of Shards

All shards contribute equal weight.

The implemented interpretation is:

- if a word is present in the local union, it counts like any other active evaluation word
- no shard gets special weighting just because it is newer, older, larger, or more local

So the current design direction is:

- equal-weight union of in-range shard words

not:

- weighted shards
- decayed shards
- priority shards

## Fitness Scale

Even though different cells may be evaluated against different local unions of words, fitness is normalized onto a
`0..1` scale.

So the intended rule is:

- local evaluation sets may differ by organism
- normalization is still required so the resulting fitness values remain bounded

This does **not** mean fitness values become perfectly comparable across every cell and generation, but it does mean
the scoring system remains bounded and selection-friendly.

## Radius Semantics

The shard coverage model is cellular and toroidal, matching the population grid.

The radius should be interpreted in the same Moore / Chebyshev sense used elsewhere in the cellular design.

So:

- radius `0` means exactly `1` cell
- radius `1` means a `3 x 3` neighborhood
- radius `2` means a `5 x 5` neighborhood

and so on until the shard effectively covers the whole toroidal population grid.

Once the radius is global, it stops growing.

## Radius Growth

New shards do not become global immediately.

The growth pattern is:

- shard is introduced at radius `0`
- after the configured growth interval, radius becomes `1`
- after the next growth interval, radius becomes `2`
- growth continues outward until the shard is effectively global

The current default is:

- grow radius every `2` generations

This is configurable through the runtime.

## Global Foundation Shard

The very first words used by the GA from the first generation onward are treated as effectively global from the
outset.

So the implemented curriculum is not “everything starts local”.

Instead:

- there is an initial globally active foundation of words
- later shard introductions use local radius growth

That gives the population a shared baseline task from generation `0`, while still allowing later curriculum additions
to diffuse through space instead of hitting the whole population at once.

## Consequences Of This Design

This spatial-shard model has a few important consequences:

- newly introduced training words stop being an immediate whole-population fitness shock
- local niches can emerge because different regions of the grid see different shard unions
- output-space growth still happens globally, so memory pressure is not avoided
- some organisms will be evaluated on more words than others
- generation-wide summary fitness becomes a rough aggregate rather than a perfectly apples-to-apples statistic

Those consequences are part of the current runtime behavior.

## Summary Of The Implemented Model

The implemented sharding model is:

- one shared global action-space catalog
- initial foundation words globally active from the outset
- later word introductions grouped into spatial shards
- shard words added to every organism’s action space immediately
- each shard assigned a cell position on the toroidal cellular grid
- each shard assigned a radius that starts local and grows outward
- local fitness-evaluation word set = union of all in-range shards
- all shards contribute equal weight
- radius growth cadence configurable, with a current preferred default of `2` generations
- shard radius stops growing once its coverage is effectively global

That is the requirements baseline for the next training-data sharding redesign.
