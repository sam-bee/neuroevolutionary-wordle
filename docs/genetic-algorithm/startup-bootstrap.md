# GA Startup Bootstrap

This document describes how `run_genetic_algorithm` bootstraps generation 0 before the first fitness evaluation.

It is about startup only. Ordinary generation advancement, spillover, and widening are mentioned only where they help
explain why startup metadata exists.

## Quick Answers

- **Is generation 0 built on the host?**
  No. Generation 0 is initialized directly in device slab slots.
- **Is there a host-side startup slab?**
  No. Startup does not materialize a temporary host slab just to upload it.
- **Is the full slab uploaded from host to device?**
  No. The host uploads the training-word catalog and configuration data, but not a whole populated slab.
- **Where does startup randomness come from?**
  A host seed. By default that seed comes from the current time in microseconds unless the CLI provides one.
- **How is device randomness produced?**
  With Philox counter-based pseudorandom streams derived from that host seed.
- **Do typical runs need identical startup populations?**
  No. The default startup path intentionally varies run to run because the default seed varies run to run.

## High-Level Goal

Startup is GPU-first.

The host still performs argument parsing, byte-budget calculations, CUDA device selection, and training-word catalog
upload. After that, the device runtime allocates an empty slab and assembles generation 0 directly into occupied slab
slots.

The old startup costs are therefore removed:

- no temporary host population of full genomes
- no second host-side slab populated by copying those genomes again
- no full-slab host-to-device upload for generation 0
- no redundant host-side `memset` passes over bytes that startup immediately overwrites

## Execution Order

### 1. Host configuration and catalog upload

The host:

- parses CLI arguments
- chooses the visible CUDA device
- loads `data/action-space-randomised.txt`
- uploads the runtime catalog to device constant memory

This stays on the host because it is small control-plane work.

### 2. Host sizing and startup-shape derivation

The host computes:

- the initial active training-word count
- the initial active action count
- the total slab byte budget
- the per-generation byte budget
- the slab slot count
- the initial population size
- the square cellular grid shape
- one startup seed

The startup seed is the only randomness the host contributes. The default seed path uses current-time microseconds. If
the CLI provides a seed, that explicit seed is used instead.

### 3. Device runtime allocation

The host allocates the device runtime buffers needed by the slab-backed GA.

Startup clears only metadata that genuinely requires a known empty state:

- slot-state metadata
- free-slot stack metadata
- generation-index arrays
- fitness and evaluation buffers
- status and summary buffers

Startup does **not** perform a full-slab payload clear just to overwrite the live startup slots immediately afterward.
Generation-0 initialization writes every live byte of every occupied startup slot directly.

### 4. Empty-slab metadata initialization on device

The device initializes the slab as empty:

- every slot is marked unoccupied
- every slot liveness count is set to `0`
- the free-slot stack is filled
- the free-slot count is set to the slab slot count

At this point the slab is valid but contains no live genomes.

### 5. Generation-0 slot allocation on device

The device allocates one slab slot per startup organism and writes those slot indices into the current-generation
buffers.

After allocation, each occupied startup slot has:

- `occupied = true`
- slot liveness count `= 1`

That count means the slot is live and owned by the current generation. Startup does **not** build any parent-use
counts, because generation 0 is not being assembled from an older parent generation.

### 6. Random genome initialization directly into occupied device slots

The runtime launches one CUDA block per organism.

Each block cooperatively initializes exactly one genome directly inside its already-allocated slab slot. Threads in the
block stride over disjoint parameter ranges so initialization uses adequate parallelism without multiple blocks racing on
the same genome bytes.

The startup kernel writes:

- input-encoder weights
- input-encoder biases
- dense-trunk weights
- dense-trunk biases
- trainable output-tail rows

The distributions match the model initializer:

- dense-layer weights use He-normal initialization
- dense-layer biases are zero
- trainable output-tail values use the configured Gaussian standard deviation

No host genome bytes are created at any point in this path.

### 7. Philox stream derivation

The runtime uses Philox because startup needs many independent random draws in parallel without locks or mutable shared
RNG state.

Each block and thread derives its pseudorandom values from a tuple of stable inputs such as:

- startup seed
- organism index
- tensor or parameter-region identifier
- parameter element index

Because Philox is counter-based, the runtime does not need:

- one shared global RNG state
- atomic coordination between threads to "draw the next random number"
- host-side random initialization followed by upload

The GPU does not act as a source of hardware entropy here. The host seed is the entropy source, and Philox expands that
seed into parallel device-side pseudorandom streams.

### 8. Generation bookkeeping finalization

Once the genomes have been written, the runtime finalizes the current-generation state:

- `current_generation_index` is `0`
- `current_generation_size` is the startup population size
- current-generation fitness fields are zeroed
- next-generation slot indices remain invalid
- next-generation fitness fields remain clear

The startup banner is printed only after generation 0 is fully live in the device slab.

### 9. First fitness evaluation

After startup completes, the runtime immediately enters the ordinary device GA path at the first fitness evaluation.

There is no special generation-0 evaluation path on the host.

## Slot Liveness Counts Versus Parent-Use Counts

Two different counting systems exist at runtime.

### Slot liveness counts

Each slab slot carries allocator metadata describing whether that slot is live.

An occupied slot keeps a slot liveness count greater than zero. During startup, a generation-0 slot simply holds a
liveness count of `1` because the current generation owns it.

This is slab-allocation metadata. It exists even before the first child-generation assembly.

### Parent-use counts

During ordinary child assembly, the runtime also builds a temporary `parent_reference_counts` array.

That array answers a different question:

- how many planned children still need parent organism `i`?

Those counts are built only after parent selection for the next generation has already happened. They are then
decremented as children are assembled so the runtime can release parent slots as soon as the final planned child has
been produced.

Startup does not use this array at all.

## Practical Consequences

- The longest silent startup phase is no longer single-threaded host random initialization.
- Startup host RAM use stays bounded because there is no full host population plus host slab resident at once.
- PCIe traffic is limited to small control data and the training-word catalog, not a full populated slab upload.
- Generation 0 begins life in exactly the slab representation used by the ordinary device runtime.
- The first real GA work after startup is device fitness evaluation, not a host-to-device population transfer.
