# `play_wordle`

`play_wordle` is the interactive inference runtime for saved winner artifacts.

It loads one saved genome into VRAM once, keeps the Wordle game loop on the host, and asks the GPU for the next guess
after each board update.

## Inputs

The runtime takes two positional arguments:

```bash
./build/play_wordle path/to/winner.bin path/to/winner.json
```

The JSON sidecar is authoritative for:

- `action_space_words`
- `action_count`
- `genome_byte_count`

`action_space_path` remains provenance only.

## Runtime shape

At startup:

1. The host reads the `.bin` and `.json` artifact pair.
2. The host validates the embedded action space and genome payload size.
3. The host uploads one genome payload and one action-word catalog to the single-model CUDA runtime.

During play:

1. The user enters a five-letter solution word or `/exit`.
2. The host validates that the supplied solution exists in the embedded action space.
3. The host owns the `WordleGrid` and applies guesses locally.
4. Before each turn, the host copies the current board to device memory.
5. One CUDA block performs policy forward plus a parallel scan across the action space and returns the best guess.
6. The host renders the updated board with ANSI tile colours.

Only one model is resident in device memory. There is no slab allocator, population runtime, or fitness evaluation
loop in this path.

## Launching With `make`

The repository `Makefile` includes a convenience target:

```bash
make play-wordle PLAY_WORDLE_MODEL=models/winner-...bin
```

By default, `PLAY_WORDLE_METADATA` is derived by replacing the `.bin` suffix with `.json`. You can override it if
needed:

```bash
make play-wordle \
  PLAY_WORDLE_MODEL=models/winner-...bin \
  PLAY_WORDLE_METADATA=models/winner-...json
```
