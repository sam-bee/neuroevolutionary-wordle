# Wordlists

This project keeps one curated Wordle action catalog and a few supporting source lists in `data/`.

## Files

- `allowed-guesses.txt`
  The full Wordle guess list. Included mainly for completeness.
- `allowed-solutions.txt`
  The 2,309 canonical solution words.
- `most-common-subtitles.txt`
  A 5,000-word subtitle-frequency source list used during action-space curation.
- `action-space.txt`
  The curated 4,739-word action space used by the policy.
- `action-space-randomised.txt`
  The same action space in randomized order. This is the runtime catalog used by the GA.

## Runtime Use

At process start, `run_genetic_algorithm` uploads the full randomized action catalog to GPU constant memory.

During a run:

- newly introduced words still come from the top of that catalog when adaptive shard-release criteria are met
- selectable action-space count and training-word count are kept equal
- output-embedding growth is therefore global
- fitness exposure is spatial:
  the initial foundation words are global, and later phases become local training-data shards on the cellular grid

For the shard schedule and spatial exposure model, see
[`training-data-sharding-curriculum.md`](training-data-sharding-curriculum.md).

For more background on how the wordlists were curated, see the [blog post on curating the
data](https://sam-burns.com/posts/neuroevolutionary-wordle-wordlists/).
