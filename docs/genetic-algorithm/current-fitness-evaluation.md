# Current Fitness Evaluation

This document describes the fitness function the project currently uses in the CUDA runtime.

It is a description of the implemented evaluation scheme, not a proposal for the next redesign.

## Scope

This is the fitness signal used when the slab-backed device runtime evaluates a population before parent selection and
next-generation assembly.

At present, the runtime uses:

- an active training-word count
- an active selectable action-space count

In the current runtime, those two counts are kept equal.

## High-Level Character

Fitness is based on simulated Wordle play.

It is not:

- a supervised loss
- a differentiable objective
- a static label-matching score

Instead, each genome is used to play Wordle episodes over the current active training-word shard, and its score is
derived from the outcomes of those episodes.

## Per-Genome Evaluation Loop

For each genome in the current generation:

1. iterate over every active training word
2. treat that word as the solution
3. run three Wordle episodes against that solution
4. sum the episode scores
5. normalize the summed score to a bounded floating-point fitness value

So if the active training-word count is `W`, each genome is evaluated on `3 * W` Wordle episodes.

## The Three Episodes Per Training Word

For each active training word, the evaluator runs:

1. one fresh-grid episode
2. one prefilled-grid episode with two fixed guesses already appended
3. one second prefilled-grid episode with a different pair of fixed guesses already appended

The current prefilled-grid setup is deterministic and shard-relative:

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

Each training word contributes three episode scores.

So:

- maximum per training word: `3 * 15 = 45`
- minimum per training word: `0`

If the active training-word count is `W`, the raw summed fitness range is:

- minimum raw fitness: `0`
- maximum raw fitness: `45 * W`

## Normalized Fitness Range

The runtime then divides the raw score by the theoretical maximum for the current word count.

That gives a normalized score in principle over `[0, 1]`.

The implementation then clamps the lower bound upward to a very small positive floor, so the actual reported and
stored range is:

- `(epsilon, 1.0]`

where `epsilon` is the project’s current positive selection-fitness floor.

So even a genome with raw score `0` does not remain at exact zero after normalization.

## What Gets Stored and Reported

The normalized fitness value is what gets:

- stored on the current generation
- used for parent selection
- aggregated into generation summaries such as best and average fitness

So the fitness values printed by `run_genetic_algorithm` are already normalized, not raw episode-score sums.

## Consequences of the Current Design

A few practical properties follow from this evaluator:

- the score depends on actual game outcomes, not proxy labels
- the score mixes virgin-grid and partially prefilled-grid play
- the score rewards faster wins but gives no graded reward for “almost solved”
- the score is normalized by the theoretical ceiling, not by empirical task difficulty
- changing the active training-word count changes the evaluation task even though the reported range stays near `0..1`

That last point matters when comparing generations across an aggressive word-count growth schedule: the values remain
normalized, but they are not guaranteed to be directly comparable as if the task difficulty were unchanged.

## Things This Fitness Function Does Not Currently Include

The current implementation does not add any explicit term for:

- novelty or diversity
- spatial niche preservation
- distance-from-solution shaping
- information gain
- repeated-guess penalties beyond whatever effect they have on winning
- separate validation-set scoring

It is a straightforward normalized episodic performance score.
