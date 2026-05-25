# Current Fitness Evaluation

This document describes the fitness function the project currently uses in the CUDA runtime.

It is a description of the implemented evaluator, not a future proposal.

## High-Level Character

Fitness is based on simulated Wordle play.

Each genome is evaluated on the GPU before next-generation assembly. The evaluator uses:

- one globally introduced action-space prefix
- one local training-word union for the genome's grid cell

The selectable action-space count is global. The evaluation word set is local.

## Per-Genome Evaluation Loop

For one genome:

1. build the union of all in-range training-data shards for that cell
2. treat each local word as the solution
3. run three Wordle episodes against that solution
4. sum the episode scores
5. normalize by the theoretical maximum for that local word count

So if a cell sees `W` local training words, that genome is evaluated on `3 * W` Wordle episodes.

## The Three Episodes Per Local Training Word

For each local training word, the evaluator runs:

1. one fresh-grid episode
2. one prefilled-grid episode with two fixed guesses already appended
3. one second prefilled-grid episode with a different pair of fixed guesses already appended

The prefilled-grid setup is deterministic and local-set-relative:

- prefilled episode A starts with guesses at offsets `entry_index + 1` and `entry_index + 2`
- prefilled episode B starts with guesses at offsets `entry_index + 3` and `entry_index + 4`
- those offsets wrap modulo the current active training-word count

This means the current fitness function deliberately evaluates more than just virgin-grid play.

## Action Selection During Evaluation

Within an episode, the genome plays deterministically.

At each step:

1. the policy model is run on the current grid state
2. every currently selectable action word is scored against the policy vector
3. the single highest-scoring action is chosen
4. that guess is appended to the grid

There is no stochastic policy sampling in the current fitness evaluation.

The action chosen is simply the argmax over the active action-space prefix.

## Episode Scoring

The current episode score is:

- `0` for a failed game
- `10 + (6 - turn_count)` for a win

So the possible episode scores are:

- loss: `0`
- win on turn 6: `10`
- win on turn 5: `11`
- win on turn 4: `12`
- win on turn 3: `13`
- win on turn 2: `14`
- win on turn 1: `15`

This means earlier wins are rewarded more strongly, but all failures collapse to zero.

## Raw Fitness Range

Each local training word contributes three episode scores.

So:

- maximum per training word: `3 * 15 = 45`
- minimum per training word: `0`

If a cell sees `W` local training words, the raw summed fitness range is:

- minimum raw fitness: `0`
- maximum raw fitness: `45 * W`

## Normalized Fitness Range

The runtime then divides the raw score by the theoretical maximum for that genome's local word count.

That gives a normalized score in principle over `[0, 1]`.

The implementation then clamps the lower bound upward to a very small positive floor, so the actual reported and
stored range is:

- `(epsilon, 1.0]`

where `epsilon` is the project's current positive selection-fitness floor.

So even a genome with raw score `0` does not remain at exact zero after normalization.

## What Gets Stored and Reported

The normalized fitness value is what gets:

- stored on the current generation
- used to rank local parent candidates
- aggregated into generation summaries such as best and average fitness

So the fitness values printed by `run_genetic_algorithm` are already normalized, not raw episode-score sums.

## Consequences of the Current Design

A few practical properties follow from this evaluator:

- the score depends on actual game outcomes, not proxy labels
- the score mixes virgin-grid and partially prefilled-grid play
- the score rewards faster wins but gives no graded reward for "almost solved"
- the score is normalized by the theoretical ceiling, not by empirical task difficulty
- different cells can be evaluated on different local word counts in the same generation
- generation summaries now aggregate across mixed local tasks

So the values remain bounded, but they are not perfectly apples-to-apples across all cells and generations.

## Things This Fitness Function Does Not Currently Include

The current implementation does not add any explicit term for:

- novelty or diversity
- spatial niche preservation
- distance-from-solution shaping
- information gain
- repeated-guess penalties beyond whatever effect they have on winning
- separate validation-set scoring

It is a straightforward normalized episodic performance score.
