# Neuroevolutionary Wordle

Requires `cmake` 3.22+ and a CUDA 13.1-compatible toolkit/driver setup. A `docker compose` development container is
included with the required dependencies if you want to work inside a container instead of on the host.

A CUDA/C++ experiment in building a Wordle-playing policy model and then training it. Genetic algorithms and
reinforcement learning will be used.

This project combines three main ideas:

1. A **neural network** that maps the current state of an in-progress Wordle game to a 5-letter word representing the
   next guess
2. A **genetic** algorithm which determines model weights
3. **Reinforcement learning**, to fine-tune the model

## High-Level Structure of the Model

The structure of the model is as follows:

1. Input encoding - turns a previous turn in a Wordle game into a 64-dimensional vector
2. Encoded input concatenation creates 321 values to pass to the main neural net
3. Main neural net - contains hidden layers of 256, 128 neurons, and a policy output head with 64 neurons
4. Output embedding - contains 64-dimensional vectors for each of 4,739 5-letter words
5. Model output selection - uses dot product method to choose a word from the action space

The current action space is **4,739 valid 5-letter guesses**. It is smaller than the full guess list, but still includes
all allowed solutions while biasing the action space toward common words derived from subtitle-frequency data.

More information on data is available at [`docs/data/data-docs.md`](docs/data/data-docs.md).
The current active-prefix baseline and intended spatial shard curriculum are described in
[`docs/data/training-data-sharding-curriculum.md`](docs/data/training-data-sharding-curriculum.md).

More detailed design information lives under the [`docs/`](docs/) folder. In particular:

- [`docs/neural-net-design/`](docs/neural-net-design/) covers the policy-model structure
- [`docs/data/`](docs/data/) covers the curated Wordle wordlists
- [`docs/genetic-algorithm/`](docs/genetic-algorithm/) covers GA design, slab-backed runtime design, and garbage
  collection

The current GA demo uploads the full 4,739-word action catalog from `data/action-space-randomised.txt` to GPU
constant memory once at process start.

During a run, `run_genetic_algorithm` still introduces new words from the top of that catalog on a configurable phased
schedule. Training-word count and selectable action-space count are kept equal, so output-embedding growth remains
global. Fitness exposure is now spatial instead of panmictic: the initial foundation words are global, later word
phases become training-data shards on the cellular grid, and each organism is evaluated against the equal-weight union
of the shards whose radius covers its cell. Recombination, mutation, and fitness evaluation stay on device. If
transient slab pressure exceeds the configured device budget, already-assembled children may spill to temporary
host-side slab storage and later be packed back into the on-device slab. Under a fixed genotype VRAM budget, the
population may shrink as active action count grows.

## Why this exists

The author has already written an older Go Wordle solver that uses shortlist reduction logic. This project is different:
the aim here is to build a model that can _learn_ a Wordle policy, while also providing opportunities to build a
non-trivial project in CUDA, to gain familiarity with neural net architecture, and to practice writing genetic
algorithms.

At a high level, the intended training story is:

- start with a neural-network policy and an output embedding
- evolve parameters with a genetic algorithm
- later add reinforcement-learning ideas if they prove useful

Development started with the **model structure first**. The repository now also includes early training-data and
genetic-algorithm runtime work, while fuller training and learning-loop design still come later.

## Current model idea

The policy model represents a Wordle game state using up to five previous turns.

Each turn consists of:

- a 5-letter guess
- 5 pieces of coloured tile feedback

Each occupied turn is passed through a **shared input encoder**. Empty turn slots do not run through the encoder; they
contribute a hard-coded 64-dimensional zero vector instead.

The model input begins with a single scalar that is `1` for a virgin grid and `0` otherwise. After that come the five
64-dimensional per-turn outputs, for a total of 321 values passed to the dense trunk.

That policy vector is scored against every word in the output embedding by dot product.

## Output embedding

Each action word has a 64-dimensional embedding vector.

- **26 dimensions are fixed** and encode per-letter counts for `A-Z` in the word
- **38 dimensions are trainable** and are initialised randomly

The network therefore does **not** emit one output per word. Instead, it emits a 64-dimensional vector in the same space
as the output embedding.

## Scope of the detailed design

For implementation detail on the neural network itself, see:

- [`docs/neural-net-design.md`](docs/neural-net-design/neural-net-design.md)
- [accompanying design diagram](docs/neural-net-design/neural-net-design-diagram.png)
- [`docs/genetic-algorithm/genetic-algorithm-design.md`](docs/genetic-algorithm/genetic-algorithm-design.md) for the
  overall character of the genetic algorithm
- [`docs/genetic-algorithm/current-fitness-evaluation.md`](docs/genetic-algorithm/current-fitness-evaluation.md) for
  the current implemented fitness function
- [`docs/genetic-algorithm/output-embedding-recombination-design.md`](docs/genetic-algorithm/output-embedding-recombination-design.md)
  for the planned output-tail recombination policy
- [`docs/genetic-algorithm/cellular-genetic-algorithm-design.md`](docs/genetic-algorithm/cellular-genetic-algorithm-design.md)
  for an experimental cellular-GA / spatial-parent-selection design
- [`docs/genetic-algorithm/genotype-slab-and-garbage-collector/genotype-slab-design.md`](docs/genetic-algorithm/genotype-slab-and-garbage-collector/genotype-slab-design.md)
  for the shared genotype slab and slab allocator design

That document is intended to be concrete enough for CUDA implementation of the model structure, while deliberately
staying focused on the policy model rather than the wider genetic algorithm, reinforcement learning, and training-loop
design.

## Repository Layout

The current documentation tree is organised as:

- [`docs/data/`](docs/data/)
  Data files, action-space notes, and training-word catalog context.
- [`docs/neural-net-design/`](docs/neural-net-design/)
  Policy-model structure and related diagrams.
- [`docs/genetic-algorithm/`](docs/genetic-algorithm/)
  High-level GA design plus slab-backed runtime and memory-management design.

The current genetic-algorithm source tree is organised as:

- [`src/genetic_algorithm/device/`](src/genetic_algorithm/device/)
  CUDA runtime orchestration, evaluation, selection, and wrapper-level slab stepping.
- [`src/genetic_algorithm/genome/`](src/genetic_algorithm/genome/)
  Dynamic genome layout and byte-interpretation helpers.
- [`src/genetic_algorithm/genotype_slab/`](src/genetic_algorithm/genotype_slab/)
  The genotype slab, slab allocator, device assembly, reference counting, and widening repacking.
- [`src/genetic_algorithm/`](src/genetic_algorithm/)
  Shared GA-level helpers such as breeding, mutation, selection, and population types.

## Build setup

The repository includes a `Makefile` with the basic project commands:

```bash
make configure
make build
make test
make test-cpu
make test-gpu
make test-gpu-sanitized
make smoke
make clean
```

For end-to-end verification after a change, `make rebuild` is the preferred command. It reformats the code, rebuilds the
project from scratch, and runs the full test suite, including the GPU-backed test.

The configure step runs the underlying CMake command:

```bash
cmake -S . -B build -G Ninja
```

## Local CUDA Device Selection

The `Makefile` will create a local `.env` file from `.env.example` the first time you run it, if `.env` does not
already exist.

The CUDA-backed executables in this project use the process environment at runtime when deciding which GPU is visible as
logical device `0`. In practice, that means the code assumes `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES` are already
set the way you want before you launch the binary.

For the `make` targets, those variables come from the local `.env` file:

```dotenv
CUDA_DEVICE_ORDER=FASTEST_FIRST
CUDA_VISIBLE_DEVICES=0
```

`make test` now includes the GPU smoke test by default. Use `make test-cpu` when you want to skip GPU coverage, and
use `make test-gpu-sanitized` to run the device smoke test under `compute-sanitizer`. When you want the strongest local
confirmation that a change is sound, use `make rebuild`.

The checked-in example assumes a **single-device machine**, where the only CUDA device is logical device `0`.

If you want to choose a different GPU on a multi-device machine, change your local `.env` file and set
`CUDA_VISIBLE_DEVICES` to the device you want. The `.env` file is machine-local and is ignored by git, so it is the
right place for per-machine CUDA selection when using `make`. If you launch `./build/run_genetic_algorithm` or another
binary directly, it will inherit whatever values your shell or profiling tool already exported.

## Status

This repository is currently at the **model implementation plus slab-backed genetic-algorithm runtime** stage.

Immediate goals:

- continue validating forward inference end-to-end
- continue hardening the slab-backed genetic-algorithm runtime
- continue refining growth handling, garbage collection, and host-spillover behaviour
- build out broader training-data and GA experiment handling
- later explore reinforcement-learning ideas if they prove useful

Broader training design still comes later.
