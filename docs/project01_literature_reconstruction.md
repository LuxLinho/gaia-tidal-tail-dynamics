# Project 01 — Literature Sample Reconstruction

## Scientific objective

Reconstruct the literature-defined Stock 2 cluster and extended/tidal-tail
candidate populations into a provenance-preserving canonical sample.

Project 01 answers:

> Which stars were selected by which literature catalogue, and how were they
> classified in the original source?

It does not determine whether those literature classifications are dynamically
correct.

## Primary catalogues

- Kos (2024)
- Risbud et al. (2025)

## Audit / indexing catalogue

- Jadhav et al. (2025)

Jadhav et al. is used as a catalogue-of-catalogues provenance and reconstruction
anchor. It is not treated as an independent third Stock 2 membership selection.

## Core principles

1. Preserve original literature provenance.
2. Preserve original membership or tail classifications.
3. Never silently remove duplicate Gaia sources.
4. Do not assign physical membership confidence in Project 01.
5. Separate catalogue reconstruction from later Gaia quality and dynamical tests.

## Planned canonical outputs

- `data/processed/stock2_literature_long.csv`
- `data/processed/stock2_literature_master.csv`
- `results/project01/project01_catalogue_summary.csv`
- `results/project01/project01_overlap_summary.csv`
- `results/project01/project01_audit.txt`

## Project 01 acceptance criteria

Project 01 is complete only when:

- Kos (2024) Stock 2 records are reproducibly reconstructed.
- Risbud et al. (2025) Stock 2 records are reproducibly reconstructed.
- Every retained record preserves its literature provenance.
- Gaia DR3 source identifiers are audited for missing and duplicated values.
- Catalogue intersection and union are explicitly quantified.
- Shared-source classification agreement/disagreement is quantified.
- No scientific reinterpretation of membership has been introduced.
