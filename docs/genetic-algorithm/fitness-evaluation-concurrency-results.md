# Fitness Evaluation Concurrency Benchmark Results

This note records the benchmark evidence behind the current CUDA fitness-evaluation concurrency selection.

## Benchmark Setup

- Command:
  `./build/run_genetic_algorithm --generations 11 --population-size 1024 --genotype-vram-gb 1 --generation-vram-gb 0.5 --initial-word-count 20 --word-count-step 1980 --shard-release-min-gap 3 --first-new-shard-release-generation 10 --shard-release-centroid-threshold 1000000 --shard-initial-radius-infinite --seed 7 --verbose`
- GPU: `NVIDIA GeForce RTX 5050 Laptop GPU`
- Driver: `580.142`
- CUDA toolkit: `13.2.78`

Generations 0 through 9 keep `action_count=20`. Generation 10 expands to `action_count=2000`.

## Selected Architecture

The chosen evaluator keeps two fitness kernels and selects between them by effective action-space size:

- for `selectable_action_count <= 128`, use the existing WMMA/tensor-core episode-tile scorer;
- for larger action spaces, use the warp-tiled direct scorer.

This preserves the fast small-generation path while avoiding the catastrophic large-generation slowdown of the pure
WMMA scorer.

## Results

| Candidate | Source | Generation 0 fitness | Large-generation fitness | Outcome |
| --- | --- | ---: | ---: | --- |
| Thread baseline | `3bd4a8c` + benchmark harness patch + local serial-selector fix | `4004.1 ms` | not completed | Capped after more than 11 minutes total wall time without finishing large-generation fitness. |
| Block-owned cooperative inference | `5d9efc0` + benchmark harness patch | `14266.9 ms` | not completed | Rejected immediately on the small benchmark. |
| Shared-action scoring without tensor cores | `c2be2d3` + benchmark harness patch | not completed | not completed | Did not finish generation-0 fitness within 3 minutes. |
| Pure WMMA scorer | instrumented `dbd0372` | `678.7 ms` | `270001.6 ms` | Good tiny-action performance, unusable on the 2000-word action space. |
| Pure warp-tiled scorer on current base | `dbd0372` with `0f5b3d8` evaluator files + final-generation timing patch | `1242.8 ms` | `4704.5 ms` | Extremely fast on the large benchmark, but a large generation-0 regression and different fitness summary than the WMMA path. |
| Scalar shared-tile scorer | local branch `candidate/shared-tile-scalar` at `1ad1186` | `660.6 ms` | not completed | Matched the small benchmark but exceeded the WMMA large-generation wall time lower bound before completion. |
| Thresholded hybrid scorer | `81e3c5c` | `676.9 ms` | `98116.9 ms` | Selected. |

## Interpretation

- Against the pure WMMA scorer, the selected hybrid improves large-generation fitness-evaluation time from
  `270001.6 ms` to `98116.9 ms`, a reduction of about `63.7%`.
- The same hybrid keeps generation-0 fitness essentially flat: `676.9 ms` versus `678.7 ms`.
- The pure warp-tiled scorer is faster again on the large generation, but it materially slows generation 0 and produces a
  different final fitness summary. It was therefore treated as a useful architecture probe rather than the final
  answer.

## Tensor-Core Notes

- Tensor cores are still used, but only on the tiny-action path.
- They were not beneficial for the 2000-word action space in the tested episode-tile structure.
- The pure WMMA design computes a fixed `16x16` score tile from only `4` active episode warps, so most of the tile
  rows are structurally idle while the large-action benchmark pays the conversion and synchronization cost repeatedly.
- The selected hybrid avoids that cost by switching the large-action case back to the warp-tiled direct scorer.

## Verification

- `dynamic_policy_test` passed.
- `fitness_evaluator_mode_test` passed.
- `fitness_evaluator_mode_test` specifically exercises both sides of the hybrid cutoff by evaluating one small
  `action_count=20` population and one large `action_count=2000` population and checking that both paths produce
  populated fitness metadata and sane summaries.

## Remaining Follow-Up

- The pure warp-tiled scorer is worth revisiting if a future change can recover its large-generation speed without
  changing the fitness summary or regressing generation 0 as badly.
- If the benchmark workload changes substantially, the `128`-action cutoff should be revalidated empirically.
