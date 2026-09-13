# Experimental benchmark

GeoAssert includes scripts for creating controlled failures in CadQuery
programs. The code supports six mutation families:

- API substitution
- Numeric parameter shift
- Boolean-operation reversal
- Syntax error
- Operation omission
- Operation reordering

Candidate mutations are checked for syntactic eligibility and, where
applicable, for a measurable difference from the reference geometry. Assertion
synthesis derives claims from executable reference programs. A filtering stage
keeps cases where the reference assertions are self-consistent and the mutated
program violates at least one assertion or fails during execution.

These mutation families provide controlled ground truth for exploratory error
localization. They are not a complete model of naturally occurring LLM errors.

Generated datasets, downloaded source corpora, raw model responses, and bulk
experiment trajectories are excluded from the public repository. Their rights,
provenance, and release readiness require separate review.
