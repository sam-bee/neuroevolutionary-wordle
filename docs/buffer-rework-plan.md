# Buffer Rework Implementation Plan

This document turns the design in `docs/buffer-rework.md` into an implementation plan tied to the current CUDA runtime.

## Purpose

The goal is to replace the current double-buffered contiguous genome storage model with a reusable arena model that:

- keeps the GA generational
- splits each organism into a fixed-size body plus fixed-size tail chunks
- supports output-vocabulary growth without resizing whole organisms
- reclaims parent storage once a parent has no remaining assigned matings
- uses less VRAM than the current full `current + next` organism buffers

## Current State

The live CUDA path still assumes that every organism is one contiguous blob:

- `src/genetic_algorithm/genome/dynamic_layout.hpp` defines `DynamicPopulationLayout` as a contiguous population buffer with one `genome_stride_bytes` per organism
- `src/genetic_algorithm/device/dynamic_runtime.hpp` still owns `current_genomes` and `next_genomes`
- `src/genetic_algorithm/device/dynamic_runtime.cu` uses `GenomePolicyModelParameters(...)` plus `GenomeTailRows(...)` everywhere for evaluation, breeding, copying, and output injection
- next-generation assembly still chooses parents inline while producing children, so there is currently no precomputed mating plan or remaining-use counter

This means the current code still pays for:

- one full current-generation organism buffer
- one full next-generation organism buffer
- whole-organism growth when action-space size increases

## Non-Goals For This Pass

The following are intentionally out of scope for the buffer rework:

- separating `training_word_count` and `action_space_word_count`
- switching to a steady-state GA
- building a general variable-sized allocator
- using linked structures or fragmented per-organism heaps
- rewriting the host-side generic fixed-capacity GA templates

The new storage design should stay explicit and regular.

## Target Runtime Model

The end state should introduce the following runtime concepts under `src/genetic_algorithm/genome/`:

- a fixed-size body slot arena for `PolicyModelParameters` and any other non-growing per-organism data
- a fixed-size tail chunk arena for output embedding rows
- a generation schema describing:
  - schema epoch
  - action count
  - tail chunk action capacity
  - tail chunk count per organism
  - action-index to `(chunk_index, in_chunk_offset)` mapping
- current-generation membership arrays that map generation indices to slot IDs
- next-generation membership arrays that map generation indices to slot IDs
- a precomputed mating plan for the whole next generation
- per-parent remaining-use counters
- arena telemetry and overflow counters

An organism should become a handle plus a schema-aware view, not a pointer to a monolithic `[body | tail]` byte slab.

## Recommended Rollout

### Phase 1: Split Representation Without Changing Allocation Strategy

Refactor the organism access layer first.

Deliverables:

- replace the current monolithic access helpers in `src/genetic_algorithm/genome/dynamic_layout.hpp`
- add explicit body and tail-chunk view types under `src/genetic_algorithm/genome/`
- add a generation schema type and helpers for mapping action index to chunk index and in-chunk offset
- keep a compatibility path that can still materialize or read a contiguous host population while the CUDA runtime is being migrated

Why this phase comes first:

- today, evaluation, breeding, copy, and injection all assume `GenomeTailRows(genome_bytes)[action_index]`
- that assumption must be removed before arena work can land cleanly

Files most affected first:

- `src/genetic_algorithm/genome/dynamic_layout.hpp`
- `src/genetic_algorithm/genome/dynamic_layout.cpp`
- `src/genetic_algorithm/device/dynamic_runtime.hpp`
- `src/genetic_algorithm/device/dynamic_runtime.cu`

Success criteria:

- runtime code no longer depends on a contiguous tail representation
- host upload and download still work
- no VRAM behavior change is required yet

### Phase 2: Introduce Arenas And Membership Arrays

Replace full-population contiguous device buffers with explicit arenas plus membership arrays.

Deliverables:

- replace `current_genomes` and `next_genomes` in `src/genetic_algorithm/device/dynamic_runtime.hpp`
- add a body arena buffer
- add a tail chunk arena buffer
- add free-slot stacks or queues for both arenas
- add current and next membership arrays of slot IDs
- add a runtime-owned schema object for current and next generations

Important constraint:

- the first version of this phase can still reserve enough slots to behave conservatively
- do not try to deliver the final VRAM savings in the same step as the storage-model rewrite

Success criteria:

- evaluation reads organisms through membership arrays and schema-aware views
- next-generation assembly writes slot IDs, not contiguous organism offsets
- download paths can reconstruct a host population for inspection and tests

### Phase 3: Add A Full Mating Plan And Remaining-Use Counters

Move parent selection out of the child-production path.

Deliverables:

- a planning pass that computes the full next-generation parent pairs before breeding starts
- per-parent use counters that include:
  - breeding uses
  - elite-copy uses if elites retain parent-owned storage semantics during assembly
- deterministic storage of the plan so breeding consumes a fixed plan rather than sampling parents inline

Why this is necessary:

- a parent can only be reclaimed safely once all of its assigned uses are known
- inline parent selection inside the assembly kernel prevents correct retirement

Files most affected:

- `src/genetic_algorithm/device/dynamic_runtime.cu`
- possibly a new planner file under `src/genetic_algorithm/genome/` or `src/genetic_algorithm/device/`

Success criteria:

- the runtime can report planned usage count per current-generation parent
- breeding consumes only precomputed parent pairs

### Phase 4: Reclaim Body Slots And Tail Chunks During Assembly

Add actual arena reuse.

Deliverables:

- decrement remaining-use counters after each completed elite copy or child production
- return body slots to the body free-slot pool when a parent hits zero remaining uses
- return all owned tail chunks to the tail free-slot pool at the same time
- reuse freed slots for subsequent children in the same generation build

Recommended implementation bias:

- start with an explicit ordered assembly pass, even if it is less parallel than today
- correctness matters more than preserving the current one-thread-per-child structure
- once reuse is correct and instrumented, more parallel planning can be revisited

Success criteria:

- peak live body slot count is lower than `current + next` full duplication in normal runs
- peak live tail chunk count is likewise reduced
- reclamation order is deterministic for a fixed seed

### Phase 5: Make Vocabulary Growth Schema-Aware

Stop treating output injection as a whole-organism resize.

Deliverables:

- introduce schema epochs
- store tail rows in fixed-size chunks
- define a fixed tail chunk action capacity for the run
- make output injection append one or more new tail chunks for the next schema epoch
- preserve existing chunk references for the old vocabulary range
- seed only the newly added output rows for injected words

Recommended assumption:

- tie tail chunk action capacity to the configured word-count growth increment unless there is a stronger reason to choose a different constant

Success criteria:

- body slots never move or resize during vocabulary growth
- action selection and inference use schema mapping instead of direct contiguous indexing
- population shrink due to whole-genome stride growth is no longer the normal mechanism

### Phase 6: Add Overflow Handling And Telemetry

The arena model needs an explicit degraded path and visibility into whether sizing assumptions are working.

Deliverables:

- runtime counters for:
  - live body slots
  - peak body slots
  - live tail chunks
  - peak tail chunks
  - body free-slot low-water mark
  - tail free-slot low-water mark
  - spill or overflow event count
  - schema-epoch transition count
- a noisy overflow path using managed memory or host-backed storage
- CLI logging in `src/cli/run_genetic_algorithm.cu`

Important constraint:

- overflow is a safety valve only
- the normal path should still fit in device arenas
- overflow events must be obvious in logs

Success criteria:

- runs emit enough information to judge whether the arena sizing heuristic is working
- overflow can be detected and diagnosed immediately

## Recommended Code Shape

The repo already moved device-side genome layout support under `src/genetic_algorithm/genome/`. This rework should continue that direction.

Recommended ownership split:

- `src/genetic_algorithm/genome/`
  - schema types
  - body slot and tail chunk view helpers
  - host-side materialization and reconstruction helpers
  - arena sizing helpers
  - telemetry structs
- `src/genetic_algorithm/device/`
  - CUDA kernels
  - device-runtime lifecycle
  - upload, download, evaluation, assembly orchestration

This keeps memory-layout logic out of the runtime orchestration layer.

## Testing Plan

The current GPU tests are anchored to the old behavior and will need to change.

Current tests that will need rewrite or replacement:

- `tests/genetic_algorithm/device/dynamic_runtime_test.cu`

That test currently expects:

- output injection to increase `genome_stride_bytes`
- population size to shrink because whole genomes got larger

Those assumptions do not survive the new design.

New tests should cover:

- schema mapping from action index to tail chunk and in-chunk offset
- correct reconstruction of a contiguous host population view from arena-backed device storage
- stable body-slot ownership across tail growth
- correct precomputed parent-use counts for a known mating plan
- reclamation of body slots when remaining-use count hits zero
- reclamation of all tail chunks owned by a retired parent
- deterministic reuse behavior for a fixed seed
- schema-epoch advancement when output embeddings are injected
- overflow-path activation and telemetry when arenas are intentionally undersized

CPU-side tests are also worth adding for:

- schema math
- slot ownership bookkeeping
- arena free-list behavior
- host reconstruction helpers

## Recommended Order Of Execution

The safest delivery order is:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6

This matches the order requested in `docs/buffer-rework.md` and avoids mixing representation, allocator, growth, and telemetry concerns into one change.

## Immediate First Slice

The first slice I would implement is Phases 1 and 2 only.

That means:

- split body and tail access into schema-aware views
- introduce arena-backed storage plus membership arrays
- keep assembly semantics simple
- postpone aggressive reuse until the storage model is stable

This is the smallest slice that creates the seam needed for the later reuse and vocabulary-growth work.
