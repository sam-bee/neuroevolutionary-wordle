# Winner Artifacts

This document describes what `run_genetic_algorithm` saves after the last generation has been evaluated.

## Save Point

At the end of the run, the CLI reads the final `PopulationFitnessSummary`, takes `best_slot_index`, downloads exactly
one winning genotype payload from device memory, and writes it to disk.

It does **not** download the whole slab.

It does **not** download the whole generation.

The device-to-host persistence transfer is:

- one summary struct
- one genotype payload of `slot_stride_bytes` bytes

## Output Directory

The winner is written to `models/` relative to the process working directory.

For the usual repo-root workflow, that means the checked-in `models/` folder.

## Filenames

The writer creates two files with the same timestamped stem:

- `winner-YYYY-MM-DD_HH-MM-SS-g<generation>-seed<seed>.bin`
- `winner-YYYY-MM-DD_HH-MM-SS-g<generation>-seed<seed>.json`

If that exact stem already exists, the writer appends `-2`, `-3`, and so on until it finds a free name.

## Binary Payload

The `.bin` file contains the raw saved genotype bytes for the winning slab slot:

- `PolicyModelParameters`
- the active trainable output-tail rows for the saved `action_count`

The payload length is written into the JSON sidecar as `genome_byte_count`.

## JSON Sidecar

The `.json` file records the metadata needed to interpret the binary payload:

- `format_version`
- `timestamp_local`
- `generation_index`
- `best_fitness`
- `best_index`
- `best_slot_index`
- `action_count`
- `genome_byte_count`
- `seed`
- `action_space_path`
- `action_space_words`

`action_space_words` is the authoritative runtime list. It contains exactly the active action-space prefix that matches
the saved genotype's output-tail rows.

`action_space_path` remains as provenance only.

## Scope

The saved winner is the best organism in the **final evaluated generation**.

It is not currently the best organism seen across the whole run.
