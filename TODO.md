tmux source-file ~/.tmux.conf
## Codex Handoff

### Goal
We are implementing "Injection of New Output Embedding" for this CUDA Wordle GA, in small steps.

The intended feature path is:
1. Preload one immutable catalog of action words to device constant memory.
2. Keep runtime training-word-count and action-space-word-count as explicit values passed into kernels that need them.
3. Use those counts to evaluate on a smaller curriculum while allowing a separately controlled active action prefix.
4. Later, inject a newly activated action on-device during next-generation assembly.
5. Later still, support true between-generation action-space growth by moving output-embedding tails out of inline genome storage.

### Why
The user wants to add newly activated playable words during evolution without downloading and re-uploading entire populations over PCIe.

The design direction is therefore:
- preload the full word catalog once
- avoid duplicate word lists in memory
- keep the business logic explicit with separate runtime counts
- keep injection device-side
- keep host uploads small
- grow only the dynamic output-embedding part, not the whole policy genome

### Completed Work
Committed work:

- `ae3f37a`
  Added synthetic hint-grid generation in `src/wordle/hint_grid.hpp` with tests in `tests/wordle/hint_grid_test.cpp`.
  Current behavior:
  - builds a 3-grid `HintGridGroup`
  - one cyclic all-yellow grid
  - two one-guess swap grids with `3 green / 2 yellow`
  - supports repeated-letter words by allowing the cyclic grid to have fewer guesses when only some rotations are valid

- `f2ff034`
  Refactored output-embedding genomes to have runtime active length inside fixed capacity.
  Main files:
  - `src/genetic_algorithm/genome.hpp`
  - `src/genetic_algorithm/breeding.hpp`
  - `src/genetic_algorithm/mutation.hpp`
  - `src/genetic_algorithm/population_initialization.hpp`
  - `src/genetic_algorithm/device/device_runtime.cu`
  - `tests/genetic_algorithm/output_embedding_genome_runtime_test.cpp`
  This means:
  - output embeddings now have `active_count`
  - breeding/mutation/evaluation respect runtime active length
  - storage is still fixed-capacity for now
  - crossover does not need further changes, because recombination is already uniform per gene

Local work in the current tree:

- added the catalog-plus-counts refactor
  Main files:
  - `src/training_folder/training_data.hpp`
  - `src/training_folder/training_data.cpp`
  - `src/training_folder/training_data.cu`
  - `src/genetic_algorithm/device/device_runtime.hpp`
  - `src/genetic_algorithm/device/device_runtime.cu`
  - `src/cli/run_genetic_algorithm.cu`
  - `tests/training_folder/training_data_test.cpp`
  - `tests/training_folder/training_data_device_test.cu`
  - `tests/genetic_algorithm/device/device_runtime_test.cu`
  This means:
  - the full 4,739-word catalog is now preloaded once to device constant memory
  - runtime evaluation now takes explicit `training_word_count` and `action_space_word_count`
  - current behavior is preserved by starting both counts at `20`

- added an isolated output-embedding injection primitive
  Main files:
  - `src/genetic_algorithm/output_embedding_injection.hpp`
  - `src/genetic_algorithm/genetic_algorithm.hpp`
  - `tests/genetic_algorithm/output_embedding_injection_test.cu`
  This means:
  - there is now a standalone `TryInjectNewOutputEmbedding(...)` helper on `ModelGenome<ActionCapacity>`
  - it seeds the new 38-float trainable tail by averaging the last 38 policy-vector dimensions over the 3 synthetic hint grids
  - it appends exactly one new active embedding when spare capacity exists
  - it is not yet wired into generation assembly or runtime count updates

Local work in the current tree:

- wired output-embedding injection into device next-generation assembly
  Main files:
  - `src/genetic_algorithm/device/device_runtime.hpp`
  - `src/genetic_algorithm/device/device_runtime.cu`
  - `src/cli/run_genetic_algorithm.cu`
  - `tests/genetic_algorithm/device/device_runtime_test.cu`
  This means:
  - `TryAssembleNextGenerationOnDevice(...)` now accepts pending output-embedding injection metadata
  - the assembly kernel injects the next catalog word into elites and children
  - failed injections now report `kOutputEmbeddingInjectionFailed`
  - the CLI now has host-side plumbing to request an injection when runtime word count is below inline capacity, and to bump both runtime counts together after a successful injected assembly
  - device-runtime coverage now includes both successful assembly-time injection and the no-spare-capacity failure path

### Revised Design Decision
The earlier "solutions are a prefix of the selectable action list" idea is superseded.

Use this instead:
- one immutable full word catalog in device constant memory
- one runtime `training_word_count`
- one runtime `action_space_word_count`

Meaning:
- training grids are created from the first `training_word_count` words
- policy action selection is limited to the first `action_space_word_count` words
- both counts may currently start equal
- both counts refer to prefixes of the same preloaded catalog

This is intentionally simpler and more explicit than carrying two separate full lists or relying on a single overloaded `entry_count`.

### Important Clarification
This design does not provide arbitrary runtime insertion of brand-new words.

It provides staged activation of words that were already uploaded in the catalog at process start.

That is acceptable for now.

True runtime growth still requires later externalization of output-embedding-tail storage, because the current genomes still store trainable tails inline and therefore still cap the number of active actions per individual.

### Current Planned Sequence
The agreed next implementation order is now:
1. Keep the isolated injection primitive and assembly integration as the low-level building blocks.
2. Add a clearer host-side policy for when injections are requested across generations.
3. Keep `training_word_count` and `action_space_word_count` equal for now.
4. Later, make that split configurable.
5. Only then externalize embedding-tail storage for true runtime growth.

### What Was Being Worked On When We Stopped
Assembly-time injection is now wired in and covered.

The next slice is higher-level host bookkeeping: deciding when to request injections across generations while keeping the two runtime counts equal for now.

### Next Concrete Slice
Add explicit host-side injection scheduling and word-count policy.

### Exact Behavioral Rules For This Slice
- Keep device-side injection itself unchanged.
- Decide an explicit host rule for when pending injection metadata is enabled.
- Keep `training_word_count == action_space_word_count` for now.
- Keep requiring active action count to stay within current inline genome capacity.

### Exact Function And Kernel Changes
Primary files to change next:
- `src/cli/run_genetic_algorithm.cu`
- tests under `tests/genetic_algorithm/device`

More concrete API direction:

- In `src/cli/run_genetic_algorithm.cu`:
  - replace the current minimal "inject whenever spare capacity exists" placeholder with an explicit policy
  - make that policy easy to inspect in logs
  - keep `training_word_count` and `action_space_word_count` equal until configurability is added
  - keep host-side bookkeeping for the next catalog word index explicit

### Follow-On Injection Slice
After host-side injection scheduling lands, the next slice should:
- support repeated injections across generations under a deliberate policy
- introduce clearer host bookkeeping for which catalog words are already active
- keep using the preloaded constant-memory catalog as the immutable source of words
- then start planning external output-tail storage for action counts beyond the current inline cap

### Verification
After implementing host-side injection scheduling:
1. Run the relevant CPU tests.
2. Run the focused GPU injection test.
3. Run the GPU-backed device-runtime and smoke tests.
4. Verify that runtime word-count changes match the intended host scheduling rule.

### Tool Status
`apply_patch` was re-tested in this session and is currently working.
