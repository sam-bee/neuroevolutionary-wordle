# Genotype Slab and Slab Allocator Design

This document describes the current shared genotype-slab design used by the genetic-algorithm runtime.

It is intended as a design overview for the storage layer and its garbage-collection behaviour. It does not attempt to
fully specify parent selection, fitness policy, or the wider training loop.

For the original design work that motivated this implementation direction, see the blog post
["Neuroevolutionary Wordle, Garbage Collection, and the Older Generation"](https://sam-burns.com/posts/neuroevolutionary-wordle-garbage-collection-and-the-older-generation/).

## Relevant Source Layout

The implementation for this design lives mainly under:

- `src/genetic_algorithm/device/`
  Runtime orchestration, fitness evaluation, parent-pair planning, and wrapper-level spill orchestration.
- `src/genetic_algorithm/genotype_slab/`
  The slab data structure itself, slot allocation, device assembly, reference counting, and growth repacking.
- `src/genetic_algorithm/genome/`
  Dynamic genome layout helpers used to interpret bytes inside slab slots.

The most relevant files are:

- `src/genetic_algorithm/device/slab_runtime.hpp`
- `src/genetic_algorithm/device/slab_runtime.cu`
- `src/genetic_algorithm/genotype_slab/slab_allocator.hpp`
- `src/genetic_algorithm/genotype_slab/slab_allocator.cpp`
- `src/genetic_algorithm/genotype_slab/device_runtime.hpp`
- `src/genetic_algorithm/genotype_slab/device_runtime.cu`
- `src/genetic_algorithm/genotype_slab/reference_counter.hpp`
- `src/genetic_algorithm/genotype_slab/repacking.hpp`

## Design Goals

The slab exists to support a GPU-first genetic algorithm where:

- genomes are large and live in global memory
- parent and child generations must briefly coexist
- genotype width can grow as the active Wordle action space grows
- the common path should stay on device
- even spill behavior should extend memory capacity rather than reimplement breeding on host
- memory use should be expressed in byte budgets rather than only in population counts

The current design explicitly prioritises correctness and bounded behaviour over maximum concurrency.

## High-Level Model

The slab is a shared fixed-width slot store for dynamic genomes.

At any moment it mainly holds:

- the current parent generation
- any still-live referenced parents during child assembly
- the next child generation being assembled
- free reusable slots

The runtime is therefore bi-generational in behaviour, but it uses one shared slab rather than two separate
population-sized genome buffers.

## Slot Layout

Each slot in the slab is fixed-width for the current action count.

That slot width is derived from the dynamic genome stride:

- policy-model parameters
- output-embedding tail rows for the current action count
- alignment padding

All slots in a given slab layout have the same width. There is no same-width compaction pass because holes are just
returned to the slab allocator and reused as ordinary free slots.

## Slab Allocator

The slab allocator is responsible for:

- the raw slot byte storage
- per-slot occupancy and slot reference state
- the free-slot stack
- slot allocation and release

The host-side structure is `HostGenotypeSlab`. Device code uses the corresponding slab views and runtime buffers.

Allocation is stack-based:

- allocating a slot pops a free slot index
- releasing a slot pushes the index back
- released slot bytes are treated as reusable storage immediately

The current implementation makes the free-list operations thread-safe and keeps slot reference counts atomic.

## Reference-Counted Garbage Collection

The current garbage collector is reference-count based.

The important references here are not arbitrary object-graph references. They are the remaining planned uses of parent
organisms during next-generation assembly.

The runtime:

1. builds parent reference counts from the child assembly plan
2. sweeps zero-reference parents before child assembly begins
3. decrements parent reference counts as children are assembled
4. releases a parent slot immediately when its final child has been assembled

This means parent reclamation is tied directly to the assembly plan and does not require a separate tracing pass for
ordinary generation turnover.

## Final-Child Priority

The runtime applies a simple pressure-reduction heuristic before child assembly.

Children are reordered so that, where possible, assembly prefers a child that is the final remaining user of one or
more parents. That tends to free parent slots earlier and reduce transient slab pressure.

This is a heuristic, not a correctness guarantee. It helps the common case but does not replace the need for either:

- enough slab slack
- or a spill/failover path

## Device Assembly

The normal assembly path is on device.

Once fitness evaluation and parent-pair planning are complete, child assembly is performed in bounded parallel CUDA
batches. The runtime does not use single-threaded child assembly, and it also does not launch unbounded work.

The device assembly path currently handles:

- plan validation
- parent reference-count construction
- zero-reference parent collection
- bounded child assembly
- parent release after each assembled batch
- cleanup on partial failure

## Growth and Repacking

When the active action count grows, genotype width increases.

That changes slot width, so the slab must be widened. The current widening path is a stop-the-world
reference-counting GC phase.

The widening repack does this:

1. preflight the widened layout
2. count referenced surviving parents
3. reject impossible growth before mutating bytes
4. compact surviving parents
5. repack them into the widened slot layout
6. rebuild slot state and free-list metadata

This phase is intentionally simple and correctness-oriented at present. It is not yet a low-latency concurrent
collector.

## Byte-Budget-Driven Sizing

The runtime sizes generations and the slab from byte budgets rather than only from explicit slot counts.

The key ideas are:

- a population byte budget
- an on-device slab byte budget
- a host spillover byte budget
- a slot width derived from the current action count
- population size derived from `population_budget / slot_stride`
- device slab slot count derived from `device_slab_budget / slot_stride`

This means population can shrink as genotype width grows, while the configured memory budget remains fixed.

The current target ratios are:

- on-device slab byte budget = `1.5 * population byte budget`
- host spillover byte budget = `0.5 * population byte budget`

So a spill event temporarily raises total available slab storage to `2.0 * population byte budget`, while keeping the
steady-state device slab smaller.

## Host Spillover

If the device slab cannot realize the next generation within its current budget, the runtime can spill already-assembled
children to host memory.

This spillover is intended to provide extra slot storage only. It is not intended to create a second host-side GA
implementation.

In particular, the host side does not need:

- a garbage collector
- widening or slot-size growth logic
- recombination logic
- mutation logic
- fitness-evaluation logic
- host-only GA code written purely so it can be unit tested without a GPU

The intended spill path is generation-scoped:

1. preserve the evaluated parent-generation summary
2. allocate a temporary fixed-width host spill slab using the configured spillover budget
3. move any children already assembled in the on-device slab into that host spill slab
4. continue child assembly on device
5. continue ordinary parent reference-counting and parent reclamation on device
6. once the full child generation exists and the parent generation has been garbage-collected, pack the spilled children
   back into the on-device slab
7. free the temporary host spill slab

So during spill:

- recombination stays on device
- mutation stays on device
- parent garbage collection stays on device
- host memory is used only as temporary child-slot storage

The CLI should still emit a loud warning when this happens so repeated spills are visible.

## Current Limitations

The current design does not yet try to solve every possible GC problem.

Notably:

- there is no same-width compaction pass because it is unnecessary for fixed-width slots
- widening repack is stop-the-world
- host spillover is an escape hatch, not the intended steady-state path
- spillover is intended to extend slab capacity only, not to mirror CUDA breeding logic on host
- there is no background collector
- there is no tracing collector for long-lived object graphs beyond the current reference-counted generation-turnover
  logic

## Current Status

For the present milestone, the slab design is intended to be:

- GPU-first on the normal path
- byte-budget driven
- reference-counted for ordinary parent reclamation
- stop-the-world for genotype-width growth
- able to spill child-slot storage to host memory when unavoidable without moving breeding or evaluation off device

That is the current design baseline from which later optimisation or more advanced garbage-collection behaviour can be
added.
