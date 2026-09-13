# Mutation-based evaluation tools

This directory contains an experimental pipeline for controlled CadQuery
program failures. It is research infrastructure, not a formally released
benchmark.

## Pipeline

1. `generation/inject_errors.py` creates eligible mutations and checks that
   they alter execution or geometry.
2. `generation/synthesize_assertions.py` derives executable assertions from a
   reference program.
3. `generation/gate_samples.py` keeps reference-consistent cases in which the
   mutation is detected.
4. `generation/build_benchmark.py` assembles feedback payloads and metadata.
5. `generation/localization.py` compares predicted and known changed steps.

Use `--smoke` on each script before a full generation run. Batch commands write
only to a user-supplied output location. Generated outputs belong under
`benchmark/generated/`, which is ignored by Git.

`samples/example_input.jsonl` is a synthetic example and is not part of a
reported experiment. `metadata/generated-sample.schema.json` describes the
core fields produced by the mutation stage.

The full builder can obtain source programs from the upstream CAD-Coder dataset.
Review upstream licensing and redistribution terms before publishing generated
data. No downloaded corpus or generated dataset is included here.
