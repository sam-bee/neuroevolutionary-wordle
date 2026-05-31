# Fitness Evaluation Benchmarking

This note is for agents evaluating performance changes to GA fitness-evaluation concurrency. The goal is to compare one
small evaluation and one large evaluation with the same executable, seed, population settings, and verbose timing output.

## Adaptive Growth Benchmark

Use the Makefile target unless you need to change the population or VRAM budgets:

```bash
make run-ga-benchmark-growth
```

The target runs:

```bash
./build/run_genetic_algorithm --generations 11 --population-size 1024 --genotype-vram-gb 1 --generation-vram-gb 0.5 --initial-word-count 20 --word-count-step 1980 --shard-release-min-gap 10 --first-new-shard-release-generation 10 --shard-release-centroid-threshold 1000000 --shard-release-fitness-p99-threshold 0.05 --shard-initial-radius-infinite --verbose
```

Generations 0 through 9 evaluate the initial 20-word training/action set. The deliberately high centroid threshold is
intended to trigger the 1,980-word shard release for generation 10, so generation 10 evaluates the expanded 2,000-word
training/action set with the new shard covering the whole population immediately. Verify this by checking that the
generation-10 summary reports `action_count=2000`.

Keep `--verbose` in benchmark commands. The current Makefile target includes it directly, and the verbose stage timing
is needed for useful fitness-evaluation comparisons.

The current Makefile target does not expose `GA_*` environment overrides. If you need a fixed seed, a different
population size, or different VRAM budgets, run `./build/run_genetic_algorithm` directly with the same options plus your
changes, for example:

```bash
./build/run_genetic_algorithm --generations 11 --population-size 1024 --genotype-vram-gb 1 --generation-vram-gb 0.5 --initial-word-count 20 --word-count-step 1980 --shard-release-min-gap 10 --first-new-shard-release-generation 10 --shard-release-centroid-threshold 1000000 --shard-release-fitness-p99-threshold 0.05 --shard-initial-radius-infinite --seed 7 --verbose
```

When comparing an optimization, record the exact command, seed, GPU, commit, CUDA driver/toolkit, and the verbose timing
lines before and after the change. The relevant comparison is not only total runtime; inspect the fitness-evaluation
stage lines separately from generation assembly, slab repacking, and artifact writing.

## One-Generation Nsight Systems Profile

Use `--generations 1` when the profile should isolate the small generation-0 fitness evaluation and avoid the second
generation entirely:

```bash
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output profiling/ga-gen0-small ./build/run_genetic_algorithm --verbose --generations 1 --population-size 1024 --genotype-vram-gb 1 --generation-vram-gb 0.5 --initial-word-count 20 --word-count-step 1980 --shard-release-min-gap 10 --first-new-shard-release-generation 10 --shard-release-centroid-threshold 1000000 --shard-release-fitness-p99-threshold 0.05 --shard-initial-radius-infinite
```

The word-count step is intentionally left in the command so the profile setup matches the adaptive growth benchmark, but
with `--generations 1` the GA stops after evaluating generation 0. No output-embedding injection or 2,000-word
generation is run.

## Evaluation Criteria

For a fitness-evaluation concurrency change, prefer evidence that shows:

- Generation 0 still reports `action_count=20`.
- The adaptive growth benchmark still reports generation 10 with `action_count=2000`.
- The verbose fitness-evaluation timing improves without shifting cost into slab preparation, parent planning, or final
  summary work.
- Nsight Systems shows better CUDA kernel occupancy, launch cadence, or GPU utilization for the generation-0 profile.
- Fitness summaries remain sane across before/after runs with the same seed; exact scores may change only if the
  optimization intentionally changes evaluation semantics.
