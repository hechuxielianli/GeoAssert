# Methodology

GeoAssert checks whether geometric claims are consistent with the states
produced by an executed CadQuery program.

## Execution model

1. Parse the Python source into an abstract syntax tree.
2. Split common linear CadQuery chains at operations that change observable
   three-dimensional geometry, such as `box`, `extrude`, `cut`, and `fillet`.
3. Execute cumulative source prefixes and retain the latest CadQuery shape.
4. Measure geometry and topology for each successful prefix.
5. Evaluate assertions against their requested step or the final state.

The splitter handles ordinary assignment-based CadQuery chains most precisely.
Control-flow statements remain executable but are treated as coarser units.

## Assertion syntax

Each assertion occupies one line:

```text
@assert [step=K | final] BODY
```

The scope defaults to `final`. Supported bodies include:

```text
volume=12000 tol=2%
bbox=(40,30,10) tol=2%
face_count=6
through_holes=1
delta_volume=decrease
is_valid=true
symmetry=XZ,YZ
volume < 0.9 * bbox_vol
```

Relational expressions are parsed as restricted Python expressions and are
evaluated only against the measured-property namespace. This restriction does
not make execution of the CAD program itself safe.

## Measurements

The verifier records volume, surface area, bounding-box dimensions and center,
topological counts, validity, and a genus-derived through-hole count. Mirror
symmetry is measured by reflecting surface samples and comparing them with the
shape surface. Windows uses a dense nearest-neighbor approximation because a
trimesh proximity path has caused native crashes with OCCT-derived meshes.

## Reports and localization

Each result records the original assertion, resolved step, pass/fail/skip
state, and a measured-value explanation. Execution errors and failed assertions
can be converted into a repair prompt. A reported failure step is evidence of
where the inconsistency becomes observable; the root cause may occur earlier.
