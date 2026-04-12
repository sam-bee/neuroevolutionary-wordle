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
1. Keep the isolated injection primitive as the low-level building block.
2. Add pending injection metadata describing which next catalog word to activate.
3. During next-generation assembly, apply the injection primitive to every child genome, including elites.
4. After a successful injection generation, have the host bump `runtime_word_counts.action_space_word_count`.
5. Only then externalize embedding-tail storage for true runtime growth.

### What Was Being Worked On When We Stopped
The isolated injection primitive now exists and passes focused tests.

The next slice is to wire that primitive into generation assembly and runtime count management.

### Next Concrete Slice
Integrate the existing injection primitive into the device GA assembly path.

### Exact Behavioral Rules For This Slice
- Add explicit pending injection metadata on the host side.
- The metadata should identify the next catalog word to activate.
- During next-generation assembly:
  - elites must receive the injected tail too
  - newly bred children must receive the injected tail too
  - every resulting genome should end the generation with `active_count + 1`
- Do not yet make the device runtime mutate persistent global counts on its own.
- After the assembly step succeeds, the host may increase `runtime_word_counts.action_space_word_count` for the next evaluation pass.
- Continue to require the active action count to stay within current inline genome capacity.

### Exact Function And Kernel Changes
Primary files to change next:
- `src/genetic_algorithm/device/device_runtime.hpp`
- `src/genetic_algorithm/device/device_runtime.cu`
- `src/cli/run_genetic_algorithm.cu`
- tests under `tests/genetic_algorithm/device`

More concrete API direction:

- In `src/genetic_algorithm/device/device_runtime.hpp`:
  - add a small pending-injection metadata struct
  - extend `TryAssembleNextGenerationOnDevice(...)` to accept that metadata

- In `src/genetic_algorithm/device/device_runtime.cu`:
  - apply `TryInjectNewOutputEmbedding(...)` inside `AssembleNextGenerationKernel(...)`
  - use the next catalog word from constant memory
  - inject after elite copy and after child breeding/mutation
  - fail the kernel cleanly if injection is requested but cannot be applied

- In `src/cli/run_genetic_algorithm.cu`:
  - decide when to request an injection
  - pass that request into next-generation assembly
  - if the injection generation succeeds, raise `runtime_word_counts.action_space_word_count` before the next evaluation

### Follow-On Injection Slice
After generation-assembly integration lands, the next slice should:
- add pending injection metadata describing which next catalog word to activate
- support repeated injections across generations
- introduce clearer host bookkeeping for which catalog words are already active
- keep using the preloaded constant-memory catalog as the immutable source of words

### Verification
After integrating injection into generation assembly:
1. Run the relevant CPU tests.
2. Run the focused GPU injection test.
3. Run the GPU-backed device-runtime and smoke tests.
4. Verify that action-space count increases only after a successful requested injection generation.

### Tool Status
`apply_patch` was re-tested in this session and is currently working.
