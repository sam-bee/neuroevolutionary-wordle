# Runtime Checkpoints

This document describes the GA runtime checkpoint boundary.

Runtime checkpoints are logical snapshots. They are not raw dumps of the reserved genotype slab allocation.

## Boundary

The checkpoint boundary is the pre-recombination boundary for generation `N + 1`:

1. generation `N` has been evaluated
2. the assembly plan for generation `N + 1` has been produced
3. slab garbage collection has released zero-reference parents using that saved assembly plan
4. no generation `N + 1` child genotypes have been allocated, recombined, mutated, or written yet

Do not describe this as checkpointing after compaction. Ordinary slab generations are non-compacting. Compaction is a
separate widening operation used when slot size changes; the ordinary checkpoint requirement is a garbage collection
pass, not a compaction pass.

## Contents

A checkpoint stores:

- schema and genome-layout versions
- validation checksum
- training-data identity hash
- generation seed and resume phase
- runtime word counts and assembly config
- pending output-embedding injection state
- runtime sizing metadata and grid state
- the evaluated current generation descriptor
- compact live genotype records, keyed by organism index
- the full saved assembly plan for generation `N + 1`

Only live genotype bytes are serialized. Free slab slots and unused reserved slab byte ranges are not serialized.

The assembly plan is authoritative on resume. It is not regenerated, and fitness evaluation and selection for generation
`N` are not rerun. Plan references are validated against the restored live organism indices before assembly resumes.

## Persistence

The device-to-host checkpoint copy is synchronous: by the time a `RuntimeCheckpoint` exists on the host, the logical live
genotype payload has been copied out of device memory.

Disk persistence writes to a temporary path first and publishes the completed checkpoint with a rename. The async writer
helper starts at most one write at a time; if a checkpoint write is still in progress when another is requested, the new
write request is rejected instead of queueing additional large snapshots.

## CLI

`run_genetic_algorithm` exposes checkpointing with:

```sh
./build/run_genetic_algorithm --checkpoint-path checkpoints/ga.bin --checkpoint-every 10
```

If `--checkpoint-path` is provided without `--checkpoint-every`, the runner checkpoints at every inter-generation
boundary. With `--verbose`, the runner prints timings for fitness evaluation, assembly-plan creation, checkpoint metadata
download, live-genotype device-to-host copy, checksum creation, async write start/finish, restore, and resume assembly.

Restore uses the saved assembly plan:

```sh
./build/run_genetic_algorithm --resume-from-checkpoint checkpoints/ga.bin --generations 100
```

On resume, the runner restores the live genotype records into a valid slab state, uploads the saved assembly plan, and
executes that plan to produce generation `N + 1`. It does not rerun generation `N` fitness evaluation, selection, or
assembly-plan creation.
