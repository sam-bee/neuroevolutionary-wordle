# Mission Statement: Fitness Evaluation Concurrency Redesign

## Goal

Redesign and benchmark the CUDA concurrency structure used by the genetic algorithm fitness evaluator so that full fitness evaluation runs materially faster, especially once the benchmark reaches the larger second-generation workload.

The target improvement is **at least 25% faster fitness evaluation time for the larger generation** compared with the current implementation, while avoiding a major slowdown in the smaller first generation. This 25% target is a useful success threshold, not a hard stopping condition. The real objective is to identify, implement, and keep the fastest sensible concurrency model among the obvious candidate architectures.

Codex should use the benchmarking procedure already documented in:

```text
docs/genetic-algorithm/fitness-evaluation-benchmarking.md
```

Benchmark results should drive the decision. Do not assume the current structure is close to optimal.

## Background

Several attempts at improving fitness-evaluation concurrency have already been made in the current branch. Treat these as useful evidence and possible implementation starting points, not as failed work to discard blindly. Codex should inspect the current branch history and working tree for prior experiments, benchmark-oriented changes, alternative kernels, abandoned structures, TODOs, and comments that may indicate what has already been tried.

These existing attempts may contain partial solutions for batching, active-game tracking, output scoring, kernel layout, or benchmark harness integration. Reuse or adapt them where they are clean and measurable. Where an existing attempt is not viable, keep the lesson from it and document why it was rejected.

The project evaluates many candidate neural-network organisms on Wordle-style training cases. Each organism is tested against many games. A game may finish in fewer than the maximum number of turns, so different organism/test-case evaluations have different amounts of remaining work as evaluation progresses.

The model architecture has these important properties:

* There is **one shared input encoder per organism**, not separate encoders for turn 1, turn 2, etc.
* The same input encoder is run over up to five previous Wordle turns for the current game.
* Empty/unavailable previous turns do not require an encoder pass; they contribute a zero vector or equivalent no-history representation.
* The encoded turn vectors are combined and passed through the dense trunk to produce a latent/policy vector.
* Action selection is performed by comparing that vector against the available output embeddings and choosing the best-scoring valid action.
* Output embeddings may be numerous, so action selection can become a large dot-product workload.

The current fitness evaluator is too slow. Earlier experiments and discussions have raised uncertainty about the correct CUDA mapping, including:

* whether one organism should be evaluated by one warp, one thread, one block, or something else;
* whether batching should be by organisms, training cases, organism/test-case pairs, Wordle turns, or output-embedding scoring work;
* whether tensor cores should be used for dense inference, output-embedding dot products, or both;
* how to avoid wasting work when some games finish before others;
* how to keep occupancy high without making the small first generation disproportionately slow.

This mission is to resolve that uncertainty empirically.

## Constraints

The implementation must preserve correctness. It is acceptable to restructure the evaluator substantially, but changes must remain testable and benchmarkable.

Codex should:

* use the existing benchmarking document as the source of truth for benchmark procedure;
* keep each significant candidate design in version control as it goes, so results can be compared and reverted;
* avoid unrelated refactors unless required to make the concurrency experiment clean;
* keep public interfaces stable where practical;
* add or update tests when changing evaluation behaviour, masking, active-game tracking, or action selection;
* document benchmark results clearly enough that the chosen architecture is justified;
* prefer concrete, measurable implementations over speculative design notes.

## Candidate Architectures to Test

Codex should try several plausible concurrency structures and benchmark them against the current baseline. The exact implementation details may be adjusted based on the existing code, but the following families should be considered.

### 1. Current structure as baseline

First establish a clean baseline using the documented benchmark process. Record timings for both the smaller first generation and the larger second generation.

This baseline is the reference point for all later changes.

### 2. Warp-per-organism/test-case evaluation

Evaluate whether assigning one warp to one active organism/test-case game remains the most sensible base unit of work.

In this model, a warp owns the state for a single active game evaluation. The warp performs the necessary encoder passes for the previous turns, the dense trunk inference, action selection, feedback application, and active/done bookkeeping.

This may be simple and divergence-tolerant, but it may underuse tensor cores and may waste work during large output-embedding scoring.

Benchmark it as a clean reference implementation if it is not already the current design.

### 3. Thread-per-organism/test-case evaluation

Test whether a simpler thread-level mapping is faster or slower than warp-level mapping for the actual model size and benchmark workload.

In this model, one thread owns one active organism/test-case game. This may improve scheduling density for small generations or small active sets, but may perform poorly for dense layers and action scoring if each thread does too much serial work.

This architecture is not expected to be ideal, but it is an important comparison point because it answers whether warp-level cooperation is actually helping.

### 4. Batch organism/test-case inference

Test a batched inference path where the unit of scheduling is a batch of active organism/test-case pairs rather than a single game.

The objective is to group active evaluations at the same stage of work and run dense layers in larger matrix-like batches. This may make better use of CUDA throughput and possibly tensor cores.

Important considerations:

* the same input encoder is reused for each populated previous turn;
* only non-empty previous turns need encoder work;
* games that have already finished should be removed from the active batch or masked cheaply;
* active batches may shrink over Wordle turns and must not leave most threads idle;
* batching overhead must not dominate the smaller first generation.

### 5. Batched output-embedding scoring

Test whether the best tensor-core target is not the dense trunk itself, but the action-selection step.

For many active organism/test-case pairs, the evaluator produces many latent/policy vectors. Each one must be scored against many output embeddings. This is naturally a matrix multiplication shape:

```text
[active policy vectors] x [output embeddings]^T -> [action scores]
```

This candidate should explore whether batching policy vectors and scoring them against the output embedding table is faster than having each warp/thread perform its own dot-product search independently.

This is especially important when the action vocabulary/output embedding count is large.

The implementation must still respect invalid-action masking, including repeated guesses and any existing legality constraints.

### 6. Hybrid model: simple inference, batched scoring

Test a hybrid approach where per-game inference remains relatively simple, but output scoring is gathered into a batched kernel or batched phase.

This may offer most of the tensor-core benefit without forcing the whole variable-turn Wordle simulation into a rigid matrix-batch structure.

This architecture may be a strong candidate if dense inference is too small or too irregular to benefit much from tensor cores, while action scoring is large and regular enough to batch effectively.

### 7. Active-set compaction or queueing between turns

Test whether maintaining a compact active set between Wordle turns improves performance.

Games finish in different numbers of turns. The evaluator should avoid spending expensive inference/scoring work on games that are already won or lost. Candidate approaches include:

* compacting active organism/test-case pairs after each turn;
* maintaining per-turn active queues;
* using cheap masks when compaction overhead would outweigh savings;
* separating small-generation and large-generation code paths if justified by benchmarks.

The chosen approach should be based on measured performance, not aesthetic preference.

## What Success Looks Like

The ideal outcome is a clear, benchmark-backed concurrency structure for the fitness evaluator.

Codex should stop only when it has one of the following:

1. a tested implementation that improves the larger second-generation benchmark by approximately 25% or more without badly regressing the smaller first generation;
2. a tested implementation that improves performance by less than 25%, but is still the fastest sensible architecture among the tested candidates and has a clear explanation of why the larger target was not reached;
3. a documented blocker showing exactly why the obvious candidate architectures could not be implemented or benchmarked reliably.

The final result should include:

* the chosen concurrency model;
* benchmark results for baseline and tested alternatives;
* the impact on the smaller first generation and larger second generation;
* notes on tensor-core use or why tensor-core use was not beneficial;
* any tests added or changed;
* any remaining follow-up work.
