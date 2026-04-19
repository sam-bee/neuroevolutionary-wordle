# Training Data Shard Curriculum

The runtime keeps the old phased word-count schedule, but no longer treats training exposure as one global active
prefix.

## What Still Works the Old Way

The following are still driven by the same phased schedule:

- `initial_word_count`
- `word_count_step`
- `word_count_step_period_generations`

New words still come from the top of `action-space-randomised.txt`, in order, and once introduced they stay
introduced.

Those introduced words are added to the action space of every organism immediately, so genotype growth is still
global.

## What Is Now Spatial

Training exposure is now spatial.

The population lives on the same square toroidal grid used for local parent selection, and each training-data phase is
represented as a shard on that grid.

Each shard has:

- a contiguous catalog range
- a center cell
- a radius
- a radius-growth cadence

The initial foundation shard is global from generation 0. Later shards start at radius 0 and expand outward.

## Local Evaluation Set

For one organism:

- find every shard whose radius covers that organism's cell
- take the union of those shard ranges
- evaluate the organism against that local union

All in-range shards contribute equal weight.

That means cells can be evaluated on different numbers of words in the same generation.

## Radius Rules

Radius uses the same toroidal Moore / Chebyshev interpretation as the spatial GA:

- radius 0: one cell
- radius 1: `3 x 3`
- radius 2: `5 x 5`

The current default is to grow shard radius every 2 generations. Once a shard is effectively global, it stops growing.

## Consequences

- New curriculum words no longer hit the entire population's fitness signal at once.
- Local niches can emerge because different regions see different shard unions.
- Memory pressure does not go away, because action-space growth is still global.
- Generation-wide best and average fitness are now aggregates over mixed local tasks, not perfectly uniform tasks.
