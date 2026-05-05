# Fitness Evaluation Benchmarking

This note is for agents evaluating performance changes to GA fitness-evaluation concurrency. The goal is to compare one
small evaluation and one large evaluation with the same executable, seed, population settings, and verbose timing output.

## Two-Generation Benchmark

Use the Makefile target unless you need to change the population or VRAM budgets:

```bash
make run-ga-benchmark-two-gen
```

The target runs:

```bash
./build/run_genetic_algorithm --generations 2 --population-size 1024 --genotype-vram-gb 1 --generation-vram-gb 0.5 --initial-word-count 20 --word-count-step 1980 --word-count-step-period 1 --shard-initial-radius-infinite --verbose
```

Generation 0 evaluates the initial 20-word training/action set. Generation 1 evaluates the expanded 2,000-word
training/action set, with the newly introduced 1,980-word shard covering the whole population immediately.

Do not set `GA_VERBOSE=0` for benchmarking. The default `GA_VERBOSE=1` makes the Makefile pass `--verbose`, which is
needed for timestamped stage progress.

Useful overrides:

```bash
GA_SEED=7 make run-ga-benchmark-two-gen
GA_BENCHMARK_POPULATION_SIZE=4096 make run-ga-benchmark-two-gen
GA_BENCHMARK_GENOTYPE_VRAM_GB=12 GA_BENCHMARK_GENERATION_VRAM_GB=6 make run-ga-benchmark-two-gen
```

When comparing an optimization, record the exact command, seed, GPU, commit, CUDA driver/toolkit, and the verbose timing
lines before and after the change. The relevant comparison is not only total runtime; inspect the fitness-evaluation
stage lines separately from generation assembly, slab repacking, and artifact writing.

## One-Generation Nsight Systems Profile

Use `--generations 1` when the profile should isolate the small generation-0 fitness evaluation and avoid the second
generation entirely:

```bash
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output profiling/ga-gen0-small ./build/run_genetic_algorithm --verbose --generations 1 --population-size 1024 --genotype-vram-gb 1 --generation-vram-gb 0.5 --initial-word-count 20 --word-count-step 1980 --word-count-step-period 1 --shard-initial-radius-infinite
```

The word-count step is intentionally left in the command so the profile setup matches the two-generation benchmark, but
with `--generations 1` the GA stops after evaluating generation 0. No output-embedding injection or 2,000-word
generation is run.

## Evaluation Criteria

For a fitness-evaluation concurrency change, prefer evidence that shows:

- Generation 0 still reports `action_count=20`.
- The two-generation benchmark still reports generation 1 with `action_count=2000`.
- The verbose fitness-evaluation timing improves without shifting cost into slab preparation, parent planning, or final
  summary work.
- Nsight Systems shows better CUDA kernel occupancy, launch cadence, or GPU utilization for the generation-0 profile.
- Fitness summaries remain sane across before/after runs with the same seed; exact scores may change only if the
  optimization intentionally changes evaluation semantics.
