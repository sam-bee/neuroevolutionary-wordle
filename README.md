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
2. Encoded input concatenation creates 320 values to pass to the main neural net
3. Main neural net - contains hidden layers of 256, 128 neurons, and a policy output head with 64 neurons
4. Output embedding - contains 64-dimensional vectors for each of 4,739 5-letter words
5. Model output selection - uses dot product method to choose a word from the action space

The current action space is **4,739 valid 5-letter guesses**. It is smaller than the full guess list, but still includes
all allowed solutions while biasing the action space toward common words derived from subtitle-frequency data.

More information on data is available at [`docs/data-docs.md`](docs/data-docs.md).

## Why this exists

The author has already written an older Go Wordle solver that uses shortlist reduction logic. This project is different:
the aim here is to build a model that can _learn_ a Wordle policy, while also providing opportunities to build a
non-trivial project in CUDA, to gain familiarity with neural net architecture, and to practice writing genetic
algorithms.

At a high level, the intended training story is:

- start with a neural-network policy and an output embedding
- evolve parameters with a genetic algorithm
- later add reinforcement-learning ideas if they prove useful

Development is starting with the **model structure only**. Training, fitness evaluation, and the genetic algorithm
machinery will come afterwards.

## Current model idea

The policy model represents a Wordle game state using up to five previous turns.

Each turn consists of:

- a 5-letter guess
- 5 pieces of coloured tile feedback

Each occupied turn is passed through a **shared input encoder**. Empty turn slots do not run through the encoder; they
contribute a hard-coded 64-dimensional zero vector instead.

The five 64-dimensional per-turn outputs are concatenated into a 320-dimensional vector, which is then processed by a
small dense trunk to produce a final 64-dimensional policy vector.

That policy vector is scored against every word in the output embedding by dot product. Repeated guesses are masked out
because they are illegal in Wordle.

## Output embedding

Each action word has a 64-dimensional embedding vector.

- **26 dimensions are fixed** and indicate whether each letter `A-Z` appears in the word
- **38 dimensions are trainable** and are initialised randomly

The network therefore does **not** emit one output per word. Instead, it emits a 64-dimensional vector in the same space
as the output embedding.

## Scope of the detailed design

For implementation detail on the neural network itself, see:

- [`docs/neural-net-design.md`](docs/neural-net-design/neural-net-design.md)
- [accompanying design diagram](docs/neural-net-design/neural-net-design-diagram.png)

That document is intended to be concrete enough for CUDA implementation of the model structure, while deliberately
stopping short of the genetic algorithm, reinforcement learning, and training-loop design.

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

That `.env` file controls which CUDA device the GPU-backed test commands run on:

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
right place for per-machine CUDA selection.

## Status

This repository is currently at the **build the model** stage.

Immediate goals:

- represent Wordle game state in a GPU-friendly form
- implement the shared input encoder
- implement the dense trunk and 64-dimensional policy head
- implement the output embedding and action scoring
- verify that forward inference works end-to-end

Training design comes later.
