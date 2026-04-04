# Neuroevolutionary Wordle

A CUDA/C++ experiment in building a Wordle-playing policy model and then training it with a genetic algorithm.

This project combines three main ideas:

1. a **neural network** that maps the current Wordle game state to a compact policy vector
2. an **output embedding** that scores candidate guesses by dot product rather than emitting one logit per action
3. a **genetic algorithm** that will eventually mutate model parameters and the trainable portion of the output embedding

The current action space is **4,739 valid 5-letter guesses**. It is smaller than the full New York Times guess list, but still includes all allowed solutions while biasing the action space toward common words derived from subtitle-frequency data.

## Why this exists

I already have an older Go Wordle solver that uses shortlist reduction logic. This project is different: the aim here is to build a model that can *learn* a Wordle policy, while also serving as an excuse to write CUDA.

At a high level, the intended training story is:

- start with a neural-network policy and an output embedding
- evolve parameters with a genetic algorithm
- later add reinforcement-learning ideas if they prove useful

Development is starting with the **model structure only**. Training, fitness evaluation, and the genetic algorithm machinery will come afterwards.

## Current model idea

The policy model represents a Wordle game state using up to five previous turns.

Each turn consists of:

- a 5-letter guess
- 5 pieces of coloured tile feedback

Each occupied turn is passed through a **shared input encoder**. Empty turn slots do not run through the encoder; they contribute a hard-coded 64-dimensional zero vector instead.

The five 64-dimensional per-turn outputs are concatenated into a 320-dimensional vector, which is then processed by a small dense trunk to produce a final 64-dimensional policy vector.

That policy vector is scored against every word in the output embedding by dot product. Repeated guesses are masked out because they are illegal in Wordle.

## Output embedding

Each action word has a 64-dimensional embedding vector.

- **26 dimensions are fixed** and indicate whether each letter `A-Z` appears in the word
- **38 dimensions are trainable** and are initialised randomly

The network therefore does **not** emit one output per word. Instead, it emits a 64-dimensional vector in the same space as the output embedding.

## Scope of the detailed design

For implementation detail on the neural network itself, see:

- [`docs/neural-net-design.md`](docs/neural-net-design.md)
- [accompanying design diagram](docs/neural-net-design-diagram.png)

That document is intended to be concrete enough for CUDA implementation of the model structure, while deliberately stopping short of the genetic algorithm, reinforcement learning, and training-loop design.

## Status

This repository is currently at the **build the model** stage.

Immediate goals:

- represent Wordle game state in a GPU-friendly form
- implement the shared input encoder
- implement the dense trunk and 64-dimensional policy head
- implement the output embedding and action scoring
- verify that forward inference works end-to-end

Training design comes later.
