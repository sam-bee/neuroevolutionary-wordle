# Neural Net Design

This document describes the agreed neural-network structure for the Wordle policy model.

It is intended as an implementation guide for the CUDA/C++ model work. The repository now also contains the CUDA
genetic-algorithm runtime, checkpointing, telemetry, winner artifacts, and interactive inference code, but the scope of
this document still ends at **model structure and forward inference**. It does **not** specify:

- genetic algorithm mechanics
- reinforcement learning
- loss functions
- fitness evaluation
- training loops
- persistence / checkpoint formats

## 1. Model overview

The model maps a Wordle game state to a **64-dimensional policy vector**.

That 64-dimensional vector is then scored against every active word in the action-space embedding by dot product.

High-level shape:

```
virgin-grid flag + up to 5 previous turns
-> shared per-turn input encoder
-> prepend virgin-grid flag to 5 encoded turn vectors
-> dense trunk
-> 64D policy output
-> dot product against output embedding
-> highest score wins
```

## 2. Action space

The full curated action catalog contains **4,739 words**.

These are all legal guesses available to the project. The list is intentionally smaller than the full NYT guess list,
while still including all solution words.

The action space size matters because the output embedding contains one vector per active action word. The GA runtime can
start from a smaller active prefix of this catalog and grow that prefix over time.

## 3. Representation of game state

A Wordle game state is represented by up to **5 previous turns**.

Each turn contains:

1. a 5-letter guess
2. 5 pieces of tile-colour feedback

The current design uses **only the current game state** as input. There is no recurrent memory and no attention mechanism.

## 4. Per-turn input representation

Each turn is encoded as **145 binary-valued features**.

### 4.1 Guess letters

Each of the 5 letter positions is represented by a 26-way one-hot vector.

So:

```math
5 * 26 = 130
```

### 4.2 Tile feedback

Each of the 5 tile positions is represented by a 3-way one-hot vector:

- green
- yellow
- grey

So:

```math
5 * 3 = 15
```

### 4.3 Total per turn

```math
130 + 15 = 145
```

## 5. Shared input encoder

Every occupied turn is processed by the same encoder weights.

This is a **shared encoder**, not five separately trained encoders.

Shape of the encoder:

```math
145 -> 128 -> 64
```

Meaning:

- input vector of length 145
- dense layer with 128 neurons
- dense layer with 64 neurons
- output is a 64-dimensional float vector

Recommended default activation:

- **ReLU** after the 128-neuron layer
- no activation on the 64-dimensional encoder output unless experimentation later suggests otherwise

### 5.1 Dense-layer interpretation

For the first encoder layer, each of the 128 neurons receives all 145 input values.

A neuron in that layer computes something of the form:

```math
h_j = f(sum_i(w_ji * x_i) + b_j)
```

Vector form:

```math
h = f(Wx + b)
```

This is a standard dense layer. The encoder is not performing lookup-table compression; it is learning a dense representation of one Wordle turn.

## 6. Empty turn handling

The model always works conceptually with 5 turn slots.

However, **empty turn slots do not run through the shared encoder**.

Instead, an empty slot contributes a hard-coded **64-dimensional zero vector**:

```math
(0, 0, ..., 0)
```

This is equivalent to treating emptiness as a special fixed encoder output, rather than passing an all-zero 145-vector through the encoder.

So per slot:

- occupied slot -> run shared encoder -> 64D vector
- empty slot -> use literal 64D zero vector

## 7. Concatenation of encoded turns

After turn encoding, there are always 5 slot outputs.

Each slot output has length 64.

These are concatenated in chronological slot order:

```math
[h1 | h2 | h3 | h4 | h5]
```

where each `h_k` is a 64-dimensional vector.

The model input also includes one scalar before those slot vectors:

- `1` if the `WordleGrid` is virgin
- `0` otherwise

Total size after prepending that scalar:

```math
1 + (5 * 64) = 321
```

Important detail:

- concatenation does **not** introduce a learned 321-neuron layer
- it simply forms one input vector of length 321 for the next dense layer

Slot order is preserved by concatenation order.

## 8. Dense trunk and policy head

After concatenation, the main network trunk is:

```math
321 -> 256 -> 128 -> 64
```

Interpretation:

- first dense hidden layer: 256 neurons
- second dense hidden layer: 128 neurons
- final policy head: 64 neurons

Recommended default activation:

- **ReLU** after the 256-neuron layer
- **ReLU** after the 128-neuron layer
- **no activation** on the final 64-dimensional policy output

### 8.1 First hidden layer equation

Let `x` be the 321-dimensional model-input vector.

A neuron in the 256-neuron layer computes:

```math
y_j = f(sum_i(w_ji * x_i) + b_j)
```

LaTeX form:

```math
y_j = f\left(\sum_{i=1}^{321} w_{ji} x_i + b_j\right)
```

Each of the 256 neurons receives **all 321 inputs**.

## 9. Output embedding

The policy head emits a **64-dimensional vector**. This is not a per-word output.

Instead, the model scores actions by comparing the policy vector with each action word’s embedding vector.

There can be up to **4,739 output-embedding vectors**, one per action word in the full catalog. Runtime genomes store
trainable tail rows for the currently active action count.

Each embedding vector has dimension 64.

### 9.1 Fixed dimensions

The first **26 dimensions** are hard-coded.

They correspond to letters `A-Z`.

For each such dimension:

- `+1` if the corresponding letter appears once in the word
- `+2` if the corresponding letter appears twice in the word
- and so on up to the actual count in the word
- `-1` if the corresponding letter does not appear in the word

This is a **count-aware** encoding over the action word itself.

Example:

- for `CRASS`, the `A` dimension is `+1`
- for `CRASS`, the `B` dimension is `-1`
- for `CRASS`, the `S` dimension is `+2`

### 9.2 Trainable dimensions

The remaining **38 dimensions** are:

- randomly initialised
- trainable
- evolved alongside the rest of the model parameters by the GA runtime

So each word embedding is:

```math
[26 fixed values | 38 trainable values]
```

## 10. Action scoring

Given:

- policy output vector `p` in `R^64`
- embedding vector `e_w` for candidate word `w`

the action score is:

```math
score(w) = dot(p, e_w)
```

For a full-catalog model, this is computed for all 4,739 action words and the highest-scoring word is selected. The
current GA and saved-artifact inference paths score the active action-space prefix associated with the runtime or saved
artifact, and dynamic inference masks words that have already been guessed on the current board.

This means the immediate inference path is:

```math
game state
-> 64D policy output
-> active-action-count dot products
-> argmax
```

## 11. Implemented parameter groups

The model contains:

### trainable model parameters

- encoder layer `145 -> 128` weights and biases
- encoder layer `128 -> 64` weights and biases
- trunk layer `321 -> 256` weights and biases
- trunk layer `256 -> 128` weights and biases
- trunk layer `128 -> 64` weights and biases
- trainable 38 dimensions for each active output embedding

### fixed data

- action-word list
- fixed 26 dimensions for each output embedding word
- hard-coded zero vector used for empty turn slots

## 12. Implemented model boundaries

The implemented model boundary covers:

1. data structures for encoded Wordle state
2. model parameter storage
3. forward pass through shared encoder
4. handling of empty turn slots
5. concatenation into 321 values
6. forward pass through dense trunk
7. output embedding scoring
8. argmax action selection

The GA runtime, artifact persistence, and interactive inference code are built around this boundary.

## 13. Summary

Current agreed model:

```
per occupied turn:
145 -> 128 -> 64

per empty turn:
64D zero vector

all 5 slots:
prepend virgin flag, then concat to 321

dense trunk:
321 -> 256 -> 128 -> 64

action selection:
dot product with active 64D word embeddings, up to 4,739 for the full catalog
highest score wins
```

This is the model structure currently implemented.
