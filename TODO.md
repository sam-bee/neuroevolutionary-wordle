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
1. Replace the current training-data shard model with a preloaded word catalog plus explicit runtime counts.
2. Make fitness evaluation use `training_word_count` and `action_space_word_count` as separate inputs.
3. Preserve current behavior by loading the full catalog once but starting with both counts at `20`.
4. Only after that, add pending injection metadata and on-device activation during next-generation assembly.
5. Only then externalize embedding-tail storage for true runtime growth.

### What Was Being Worked On When We Stopped
The old next slice was an action-vs-solution split using a single shared prefix with two counts stored inside `TrainingDataShard`.

Do not implement that old design now.

The new next slice is the catalog-plus-counts refactor described below.

### Next Concrete Slice
Implement a single immutable word catalog in constant memory and pass explicit runtime counts into the device runtime.

Suggested data-model changes:

- In `src/training_folder/training_data.hpp`, replace the current `TrainingDataShard` shape with something conceptually like:

```cpp
constexpr std::size_t kTrainingWordCatalogCapacity = 4739;

struct TrainingWordCatalog {
    common::FixedBuffer<wordle::Word, kTrainingWordCatalogCapacity> words{};
    std::size_t word_count = 0;
};
```

- Keep the existing phased-curriculum idea, but make it operate on an explicit configured training count, not on the catalog object itself.

- Add a small runtime-counts struct in the device runtime, conceptually like:

```cpp
struct RuntimeWordCounts {
    std::size_t training_word_count = training_folder::kTrainingDataCurriculumEntryCount;
    std::size_t action_space_word_count = training_folder::kTrainingDataCurriculumEntryCount;
};
```

### Exact Behavioral Rules For This Slice
- Upload the full curated action-space catalog once from `data/action-space-randomised.txt`.
- Require `word_count <= kTrainingWordCatalogCapacity`.
- Require `training_word_count <= word_count`.
- Require `action_space_word_count <= word_count`.
- For now, require `action_space_word_count <= kDeviceActionCount`, because genomes still only have inline tail capacity for the current device action count.
- Preserve existing behavior by initially setting both runtime counts to `kTrainingDataCurriculumEntryCount`.
- Continue phased curriculum behavior by evaluating:
  - generations `0-99` on only the first `kTrainingDataEntriesPerShard` training words
  - generation `100+` on all configured `training_word_count` words

### Exact Function And Kernel Changes
Primary files to change:
- `src/training_folder/training_data.hpp`
- `src/training_folder/training_data.cpp`
- `src/training_folder/training_data.cu`
- `src/genetic_algorithm/device/device_runtime.hpp`
- `src/genetic_algorithm/device/device_runtime.cu`
- `src/cli/run_genetic_algorithm.cu`
- tests under `tests/training_folder` and `tests/genetic_algorithm/device`

More concrete API direction:

- In `src/training_folder/training_data.hpp` and `.cpp`:
  - replace the top-20 loader with a full-catalog loader
  - rename the loader and uploader APIs accordingly
  - keep a helper for default path resolution
  - keep validation on the host and device sides

- In `src/training_folder/training_data.cu`:
  - store the full catalog in one `__constant__` symbol
  - expose `DeviceTrainingWordCatalog()`

- In `src/genetic_algorithm/device/device_runtime.hpp`:
  - add `RuntimeWordCounts`
  - change `TryEvaluatePopulationFitnessOnDevice` to take `const RuntimeWordCounts &`
  - do not add device-global mutable counts for this slice

- In `src/genetic_algorithm/device/device_runtime.cu`:
  - rename helpers from shard terminology to catalog terminology
  - change `TryEvaluateIndividualFitness(...)` to take `generation_index` plus `RuntimeWordCounts`
  - change `TryInitializePrefilledGrid(...)` to wrap within the active training-word count
  - build selectable action embeddings from the first `action_space_word_count` catalog words
  - clamp selectable action count against genome active embedding count
  - change `EvaluatePopulationFitnessKernel(...)` to receive the two runtime counts as kernel parameters
  - leave summary kernel unchanged
  - leave next-generation assembly unchanged for this slice

- In `src/cli/run_genetic_algorithm.cu`:
  - load the full catalog once
  - upload it once to constant memory
  - create `RuntimeWordCounts runtime_word_counts{}`
  - initialize both counts to `kTrainingDataCurriculumEntryCount`
  - pass those counts into `TryEvaluatePopulationFitnessOnDevice(...)`
  - update logging to print catalog size, configured training count, and configured action count separately

### Why Counts Should Be Kernel Parameters
The user suggested device-global counts. For the current slice, prefer passing counts as kernel parameters instead.

Reasons:
- the host is already launching the kernels
- the values change rarely
- parameter passing is simpler and easier to reason about
- it avoids introducing mutable device-global state before injection logic actually needs it

If later on-device assembly needs to produce a new active action count, that can be added in the injection slice with explicit metadata rather than introduced prematurely here.

### Follow-On Injection Slice
After the catalog-plus-counts refactor lands, the next injection slice should:
- add pending injection metadata describing which next catalog word to activate
- during next-generation assembly, seed the new output-embedding tail from synthetic hint grids
- raise the child genomes' active output-embedding count accordingly
- ensure elites also receive the injected tail

That slice should continue to read the newly activated word from the preloaded constant-memory catalog.

### Verification
After implementing the catalog-plus-counts refactor:
1. Run the relevant CPU tests.
2. Run GPU-backed training-data and device-runtime tests if available in the environment.
3. Verify that current runtime behavior is unchanged when both counts start at `20`.

### Tool Status
`apply_patch` was re-tested in this session and is currently working.
