# Reproduction

## Supported environment

The verified public path uses Python 3.11 and CadQuery 2.7 from conda-forge.
Create the environment and install the local package without replacing the
conda-provided geometry stack:

```bash
conda env create -f environment.yml
conda activate geoassert
python -m pip install -e . --no-deps
python -m pytest
python examples/basic_verification.py
```

## Benchmark smoke checks

Install the optional download dependency only when rebuilding from the source
corpus:

```bash
python -m pip install huggingface-hub
python benchmark/generation/localization.py --smoke
python benchmark/generation/inject_errors.py --smoke
python benchmark/generation/synthesize_assertions.py --smoke --no-symmetry
python benchmark/generation/gate_samples.py --smoke --no-symmetry
python benchmark/generation/build_benchmark.py --smoke --no-symmetry
```

The full generator can download an upstream CAD-Coder data file through
`huggingface_hub`. Review the upstream dataset terms before redistributing any
derived dataset. The public repository does not ship that cache or generated
benchmark outputs.

## Determinism and platform notes

Surface sampling uses deterministic seeds where the benchmark requires stable
comparisons. Native OpenCASCADE behavior and tessellation can still differ
between dependency or platform versions. Record Python and package versions
when producing results.
