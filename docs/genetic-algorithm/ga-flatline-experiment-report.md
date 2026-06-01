# GA Flatline Experiment Report

Date: 2026-06-02

## Scope

This report summarizes five reduced-population experiment branches created to investigate the GA flatlining seen after curriculum/action-space expansion. The objective was not to reproduce production scale exactly. All experiment runs used deliberately lower memory allowances than `make run-ga-prod` implies, which gives much smaller population sizes and faster turnaround. That is often desirable for testing, but the results should be validated again at larger population sizes before changing production defaults.

Unless noted otherwise, runs used:

- Seed: `123`
- Generations: `100` or `180`
- Population ceiling: `1024`
- Genotype VRAM: `0.7 GiB`
- Generation VRAM: `0.45 GiB`
- Initial words: `50`
- Word step: `50`
- Breeding radius: `3`
- Parent selection rank exponent: `0.5`
- Baseline crossover temperatures: `0.02`, `0.01`, `0.005`
- Telemetry: `telemetry/experiments/*.sqlite`, ignored by git

## Branches

| Branch | Commit | Hypothesis | Validation |
| --- | --- | --- | --- |
| `experiment/p99-gated-curriculum` | `6cad06b` | Convergence-only shard release is causing runaway expansion before competence. | Strongly validated as a runaway-expansion cause, but not enough alone to reach larger lexicons. |
| `experiment/stratified-shard-telemetry` | `dfa71cd` | Aggregate p99/median hides local shard exposure and action-space shock. | Strongly validated as diagnostic evidence. |
| `experiment/global-shard-release` | `6956b1c` | Spatially staged shard exposure is the main cause of the cliff. | Partially validated. It reduces early shock, but later convergence collapse remains. |
| `experiment/shaped-loss-fitness` | `61b74d6` | Win-only fitness is too sparse after action expansion; loss progress should guide selection. | Strongly positive, with the important caveat that the metric scale changed. |
| `experiment/mutation-only-breeding` | `ec5b9e3` | Recombination is destroying useful partial policies; mutation-only breeding may recover better. | Falsified in this test. It made over-expansion worse. |

## Baseline Failure

The adaptive-curriculum baseline reproduced the failure with reduced population size.

| Generation | Words | Mean | p99 | Max |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 50 | 0.0758 | 0.1111 | 0.1516 |
| 20 | 50 | 0.1400 | 0.1596 | 0.1653 |
| 21 | 100 | 0.0222 | 0.0382 | 0.0542 |
| 38 | 150 | 0.0150 | 0.0396 | 0.0453 |
| 69 | 250 | 0.0065 | 0.0179 | 0.0191 |
| 147 | 600 | 0.0073 | 0.0090 | 0.0093 |
| 177 | 750 | 0.0031 | 0.0062 | 0.0070 |
| 179 | 750 | 0.0035 | 0.0067 | 0.0073 |

The baseline gets a good logarithmic improvement on the initial 50-word problem, then expands due to convergence and never recovers. By generation 177 it is evaluating 750 words with p99 near zero.

## Findings

### 1. P99-Gated Curriculum

Disabling convergence-triggered releases stopped runaway expansion. The run released 50 -> 100 words at generation 25 on a real p99 trigger and never released beyond 100 words by generation 179.

| Generation | Words | Mean | p99 | Max |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 50 | 0.1400 | 0.1596 | 0.1653 |
| 25 | 100 | 0.0224 | 0.0333 | 0.0449 |
| 50 | 100 | 0.0521 | 0.0644 | 0.0773 |
| 100 | 100 | 0.0676 | 0.0716 | 0.0718 |
| 179 | 100 | 0.0800 | 0.0849 | 0.0849 |

Conclusion: convergence-only release is a real bug in the curriculum policy. It mistakes genetic collapse for competence. However, this branch only proves that gating prevents over-expansion; it does not prove high fitness at larger lexicons because it stayed at 100 words.

### 2. Stratified Shard Telemetry

The diagnostic table grouped organisms by their local training word exposure. It showed that the generation-21 cliff is not just caused by organisms newly exposed to the new shard. At generation 21, 991 of 992 organisms still had local exposure of 50 words, but their p99 had already collapsed because the global selectable action count had expanded to 100.

| Generation | Local words | Organisms | Mean | p99 | Max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 50 | 1024 | 0.1400 | 0.1596 | 0.1653 |
| 21 | 50 | 991 | 0.0222 | 0.0382 | 0.0542 |
| 21 | 100 | 1 | 0.0422 | 0.0422 | 0.0422 |
| 38 | 50 | 702 | 0.0153 | 0.0396 | 0.0453 |
| 38 | 100 | 290 | 0.0143 | 0.0347 | 0.0371 |

Conclusion: the first cliff is primarily global action-space expansion shock. The spatial shard wave also matters, but it is not the whole explanation. This strongly supports adding permanent telemetry that separates global action count, local exposure, and case-level performance.

### 3. Global Shard Release

Making new shards cover the whole grid immediately improved the first shock and delayed the next release.

| Generation | Words | Mean | p99 | Max |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 50 | 0.1400 | 0.1596 | 0.1653 |
| 21 | 100 | 0.0400 | 0.0496 | 0.0518 |
| 44 | 100 | 0.0607 | 0.0691 | 0.0718 |
| 45 | 150 | 0.0269 | 0.0320 | 0.0342 |
| 77 | 250 | 0.0134 | 0.0179 | 0.0202 |
| 99 | 350 | 0.0115 | 0.0138 | 0.0141 |

Conclusion: spatial staging is harmful, but removing it is not sufficient. It improves the first expansion and delays over-expansion, then convergence collapses again and releases resume at the minimum gap.

### 4. Shaped Loss Fitness

This branch gave losing episodes a small progress reward based on the best feedback seen, while keeping all wins much more valuable than losses. It produced the strongest positive result.

| Generation | Words | Mean | p99 | Max |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 50 | 0.1147 | 0.1489 | 0.1877 |
| 20 | 50 | 0.1698 | 0.1924 | 0.1949 |
| 22 | 100 | 0.0661 | 0.0948 | 0.0987 |
| 38 | 100 | 0.1039 | 0.1430 | 0.1525 |
| 60 | 150 | 0.0696 | 0.0783 | 0.0811 |
| 70 | 200 | 0.0570 | 0.0713 | 0.0744 |
| 80 | 250 | 0.0536 | 0.0634 | 0.0664 |
| 99 | 250 | 0.0622 | 0.0715 | 0.0777 |

Conclusion: the sparse win-only fitness signal is probably a major cause of failure. This branch preserved useful selection signal after expansion and avoided racing past 250 words by generation 100. Caveat: the score definition changed, so these values are not directly comparable to old win-only fitness. The next implementation should log both shaped selection fitness and old win-only fitness/win rate.

### 5. Mutation-Only Breeding

Setting all crossover temperatures to zero made the system worse in this test.

| Generation | Words | Mean | p99 | Max |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 50 | 0.1423 | 0.1556 | 0.1600 |
| 17 | 100 | 0.0188 | 0.0382 | 0.0398 |
| 33 | 150 | 0.0099 | 0.0267 | 0.0296 |
| 54 | 250 | 0.0072 | 0.0207 | 0.0225 |
| 75 | 350 | 0.0099 | 0.0122 | 0.0132 |
| 95 | 450 | 0.0095 | 0.0115 | 0.0116 |
| 99 | 450 | 0.0101 | 0.0116 | 0.0124 |

Conclusion: disabling crossover is not supported. It accelerated genetic convergence and caused even faster curriculum over-expansion. This does not prove the current recombination strategy is optimal, but it does falsify the simple "crossover is the main destroyer" version of the hypothesis.

## Literature Context

The results line up with several known evolutionary-learning issues:

- Curriculum learning: Bengio et al. describe the benefit of gradually increasing task difficulty. The GA data supports this general idea, but shows the implementation must be competence-gated and must avoid changing the action space faster than selection can adapt. Source: <https://icml.cc/2009/papers/119.pdf>
- Deceptive objectives and novelty: Lehman and Stanley show that objective-only search can be misled by deceptive gradients. The convergence-triggered release rule is a concrete local example: low genetic diversity is being interpreted as readiness, when it can instead mean collapse. Source: <https://pubmed.ncbi.nlm.nih.gov/20868264/>
- Lexicase and case-aware selection: epsilon-lexicase selection preserves individuals that are good on different cases rather than reducing everything to one aggregate score. The stratified telemetry result suggests this system needs at least case/exposure-aware telemetry and probably case-aware parent selection experiments. Source: <https://gpbib.cs.ucl.ac.uk/gp-html/LaCava_2016_GECCO.html>
- Quality diversity: MAP-Elites preserves high-performing elites across behavior descriptors instead of a single global elite. That is relevant because the grid currently collapses genetically while evaluation conditions differ by shard exposure. Source: <https://arxiv.org/abs/1504.04909>

## Recommendations

1. Keep convergence telemetry, but do not let convergence alone release shards.
   The p99-gated branch strongly validates this. Use convergence collapse as a warning signal or diversity trigger, not as evidence that a new shard should be released.

2. Move shaped loss fitness forward, but add dual telemetry.
   The shaped branch is the strongest positive result. Selection should probably use a shaped score, while telemetry should also log old win-only fitness and win rate so production success remains measurable.

3. Treat action-space expansion shock as a first-class problem.
   The stratified telemetry showed the action-space increase hurts organisms even before they locally train on the new shard. Test local action masks, gradual action admission, or separate training-word release from selectable-action release.

4. Add permanent stratified and case-level telemetry before interpreting aggregate p99.
   Aggregate p99, mean, and median are not enough. At minimum, log global action count, local training exposure, per-shard exposure, old win-only score, shaped score, and win rate.

5. Do not pursue mutation-only breeding as the fix.
   It made convergence and over-expansion worse. Recombination may still need better structure, but disabling crossover is not supported by this evidence.

## Next Experiment

The best next branch should combine:

- p99-gated shard release, with convergence release disabled.
- shaped selection fitness.
- dual score telemetry: shaped score, old win-only score, and win rate.
- optional global shard exposure or action-mask/ramp, tested separately.

Run it first with the same reduced-memory setup, then repeat with larger memory/population settings closer to `make run-ga-prod` only after the reduced run shows old-metric win-rate improvement at 250+ words.
