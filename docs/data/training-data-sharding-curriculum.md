# Training Data Shard Curriculum

The runtime grows the training/action catalog adaptively. Training exposure is spatial rather than one global active
prefix.

## What Still Works By Count

The following still define the size of the curriculum:

- `initial_word_count`
- `word_count_step`

New words still come from the top of `action-space-randomised.txt`, in order, and once introduced they stay
introduced.

Those introduced words are added to the action space of every organism immediately, so genotype growth is still
global.

## Shard Release Criteria

The initial foundation shard is global from generation 0. The first later shard cannot release before generation 10.
After that, later shards are released only after the configured minimum gap and when either adaptive trigger fires:

- no shard has been released in the last 10 generations
- p99 fitness is at least 0.05 higher than it was at the previous shard release
- centroid distance mean dips below 4

The CLI exposes `--first-new-shard-release-generation N`, `--shard-release-min-gap N`, and
`--shard-release-centroid-threshold F`, and `--shard-release-fitness-p99-threshold F`. The fitness threshold flag is a
required p99 gain over the previous release baseline, not an absolute p99 floor. The first new-shard release generation
may not be set below 10, and the minimum gap may not be set below 10. Each release writes a console line with the
inserted catalog range and whether fitness evaluations, convergence, or both triggered it.

## What Is Now Spatial

Training exposure is now spatial.

The population lives on the same toroidal grid used for local parent selection. Startup grids are square, but later
grids may be rectangular after row-only population shrink. Each training-data phase is represented as a shard on that
grid.

Each shard has:

- a contiguous catalog range
- a two-dimensional center coordinate
- an initial radius
- a radius
- a radius-growth cadence

Later shards default to radius 0 when they are actually released and expand outward from that release generation.

Shard centers are assigned against the original startup grid. If later row deletion removes a shard center, its row is
clamped to the last surviving row and its column is preserved.

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

The current default is to grow shard radius every 2 generations.

The runtime also supports a configurable initial radius for newly introduced non-foundation shards. The
`run_genetic_algorithm` CLI exposes `--shard-initial-radius N` and `--shard-initial-radius-infinite`; the infinite flag
sets the shard radius high enough to cover the whole population grid immediately.

## Consequences

- New curriculum words no longer hit the entire population's fitness signal at once.
- Local niches can emerge because different regions see different shard unions.
- Memory pressure does not go away, because action-space growth is still global.
- Generation-wide best and average fitness are now aggregates over mixed local tasks, not perfectly uniform tasks.
