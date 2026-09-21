# Project 06 verification

- Full suite after README update: **80 tests PASS**, zero failures/errors/skips.
- 14 new Project 06 tests; 66 existing tests pass.
- Command: `MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v`.
- Complete log: `test_results.txt`; `git diff --check` passed.
- Four independent stages executed successfully and ECSV products read back.
- 294 upstream files match their pre-execution SHA256 values.
- 26 tracked Project 05 scientific code/documentation/result files match HEAD byte-for-byte.
- Canonical 1456 unique sources, row order, C/L/T and catalogue labels unchanged;
  all 1456 inherited 3D positions remain calculable. Frozen Project 05 reference/orbit unchanged.
- Native galpy midplane rtide and the analytic point-mass limit independently
  validate the Jacobi formula and unit handling; denominator identities pass.
- Six PNG figures visually reviewed: axes/units, current point, scale guides,
  full candidate range, legends, and no misleading membership boundary. PDF pairs generated.
- All required mass, field, radius, counts, orbital and output checks PASS.
- Mass is an approximate user-specified literature baseline; primary full-text
  derivation not independently audited. No formal uncertainty invented.
- Local circular-equivalent midplane approximation is explicit; off-plane
  forces are auxiliary and never mixed into the midplane denominator.
- Intermediate ECSV outputs follow existing Git ignore rules and are locally reproducible.

## Existing files modified

- `README.md`: Project 06 tidal-field/Jacobi stage COMPLETE; Project 07+ unchanged.
- `tests/test_project05_sensitivity.py`: retain reference/orbit/population hashes;
  treat the historical README-unchanged flag as historical rather than banning
  authorized future README edits. No Project 05 science changed.

## New files

- `data/interim/project06/stock2_candidate_tidal_zone_counts.ecsv`
- `data/interim/project06/stock2_candidate_tidal_zones.ecsv`
- `data/interim/project06/stock2_jacobi_mass_sensitivity.ecsv`
- `data/interim/project06/stock2_jacobi_radius.ecsv`
- `data/interim/project06/stock2_orbital_jacobi_radius.ecsv`
- `data/interim/project06/stock2_tidal_field.ecsv`
- `docs/project06_tidal_field_jacobi_radius.md`
- `results/project06/candidate_extent/candidate_xy.pdf`
- `results/project06/candidate_extent/candidate_xy.png`
- `results/project06/candidate_extent/class_radial_distribution.pdf`
- `results/project06/candidate_extent/class_radial_distribution.png`
- `results/project06/candidate_extent/radial_distribution.pdf`
- `results/project06/candidate_extent/radial_distribution.png`
- `results/project06/jacobi_radius/mass_sensitivity.pdf`
- `results/project06/jacobi_radius/mass_sensitivity.png`
- `results/project06/orbital_variation/jacobi_R.pdf`
- `results/project06/orbital_variation/jacobi_R.png`
- `results/project06/orbital_variation/jacobi_time.pdf`
- `results/project06/orbital_variation/jacobi_time.png`
- `results/project06/orbital_variation/orbital_variation_summary.json`
- `results/project06/project06_metadata.json`
- `results/project06/project06_summary.txt`
- `results/project06/test_results.txt`
- `results/project06/tidal_field/metadata.json`
- `results/project06/upstream_integrity.json`
- `results/project06/verification.md`
- `src/project06/01_tidal_field.py`
- `src/project06/02_jacobi_radius.py`
- `src/project06/03_candidate_extent.py`
- `src/project06/04_orbital_variation.py`
- `src/project06/support.py`
- `tests/test_project06_tidal_scale.py`

No commit, staging or push performed.
