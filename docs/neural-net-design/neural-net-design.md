# Neural Net Design

This document describes the agreed neural-network structure for the Wordle policy model.

It is intended as an implementation guide for the initial CUDA/C++ work. The scope of this document ends at **model structure and forward inference**. It does **not** yet specify:

- genetic algorithm mechanics
- reinforcement learning
- loss functions
- fitness evaluation
- training loops
- persistence / checkpoint formats

## 1. Model overview

The model maps a Wordle game state to a **64-dimensional policy vector**.

That 64-dimensional vector is then scored against every word in the action-space embedding by dot product.

High-level shape:

```text
up to 5 previous turns
-> shared per-turn input encoder
-> concatenate 5 encoded turn vectors
-> dense trunk
-> 64D policy output
-> dot product against output embedding
-> mask illegal repeated guesses
-> highest remaining score wins
```

## 2. Action space

The action space contains **4,739 words**.

These are all legal guesses for the model to play. The list is intentionally smaller than the full NYT guess list, while still including all solution words.

The action space size matters because the output embedding contains one vector per action word.

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

```text
5 * 26 = 130
```

### 4.2 Tile feedback

Each of the 5 tile positions is represented by a 3-way one-hot vector:

- green
- yellow
- grey

So:

```text
5 * 3 = 15
```

### 4.3 Total per turn

```text
130 + 15 = 145
```

## 5. Shared input encoder

Every occupied turn is processed by the same encoder weights.

This is a **shared encoder**, not five separately trained encoders.

Shape of the encoder:

```text
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

```text
h_j = f(sum_i(w_ji * x_i) + b_j)
```

Vector form:

```text
h = f(Wx + b)
```

This is a standard dense layer. The encoder is not performing lookup-table compression; it is learning a dense representation of one Wordle turn.

## 6. Empty turn handling

The model always works conceptually with 5 turn slots.

However, **empty turn slots do not run through the shared encoder**.

Instead, an empty slot contributes a hard-coded **64-dimensional zero vector**:

```text
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

```text
[h1 | h2 | h3 | h4 | h5]
```

where each `h_k` is a 64-dimensional vector.

Total size after concatenation:

```text
5 * 64 = 320
```

Important detail:

- concatenation does **not** introduce a learned 320-neuron layer
- it simply forms one input vector of length 320 for the next dense layer

Slot order is preserved by concatenation order.

## 8. Dense trunk and policy head

After concatenation, the main network trunk is:

```text
320 -> 256 -> 128 -> 64
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

Let `x` be the concatenated 320-dimensional vector.

A neuron in the 256-neuron layer computes:

```text
y_j = f(sum_i(w_ji * x_i) + b_j)
```

LaTeX form:

```text
y_j = f\left(\sum_{i=1}^{320} w_{ji} x_i + b_j\right)
```

Each of the 256 neurons receives **all 320 inputs**.

## 9. Output embedding

The policy head emits a **64-dimensional vector**. This is not a per-word output.

Instead, the model scores actions by comparing the policy vector with each action word’s embedding vector.

There are **4,739 output-embedding vectors**, one per action word.

Each embedding vector has dimension 64.

### 9.1 Fixed dimensions

The first **26 dimensions** are hard-coded.

They correspond to letters `A-Z`.

For each such dimension:

- `+1` if the corresponding letter appears at least once in the word
- `-1` if the corresponding letter does not appear in the word

This is a **presence-only** encoding. It does not encode counts.

Example:

- for `CRASS`, the `A` dimension is `+1`
- for `CRASS`, the `B` dimension is `-1`

### 9.2 Trainable dimensions

The remaining **38 dimensions** are:

- randomly initialised
- trainable
- updated alongside the rest of the model parameters once training code exists

So each word embedding is:

```text
[26 fixed values | 38 trainable values]
```

## 10. Action scoring

Given:

- policy output vector `p` in `R^64`
- embedding vector `e_w` for candidate word `w`

the action score is:

```text
score(w) = dot(p, e_w)
```

The model computes this for all 4,739 action words.

Then:

1. mask out repeated guesses, because they are illegal
2. select the legal word with the highest remaining score

This means the immediate inference path is:

```text
game state
-> 64D policy output
-> 4,739 dot products
-> masking
-> argmax
```

## 11. Parameters that exist at model-build stage

For the initial implementation, the model contains:

### trainable model parameters

- encoder layer `145 -> 128` weights and biases
- encoder layer `128 -> 64` weights and biases
- trunk layer `320 -> 256` weights and biases
- trunk layer `256 -> 128` weights and biases
- trunk layer `128 -> 64` weights and biases
- trainable 38 dimensions for each of the 4,739 output embeddings

### fixed data

- action-word list
- repeated-guess legality mask input for inference
- fixed 26 dimensions for each output embedding word
- hard-coded zero vector used for empty turn slots

## 12. Suggested implementation boundaries

For now, the implementation target should stop at:

1. data structures for encoded Wordle state
2. model parameter storage
3. forward pass through shared encoder
4. handling of empty turn slots
5. concatenation into 320 values
6. forward pass through dense trunk
7. output embedding scoring
8. repeated-guess masking
9. argmax action selection

That is enough to validate the architecture and test inference.

## 13. Summary

Current agreed model:

```text
per occupied turn:
145 -> 128 -> 64

per empty turn:
64D zero vector

all 5 slots:
concat to 320

dense trunk:
320 -> 256 -> 128 -> 64

action selection:
dot product with 4,739 64D word embeddings
mask repeated guesses
highest legal score wins
```

This is the model structure to implement first.

Everything else can be built around it later.
