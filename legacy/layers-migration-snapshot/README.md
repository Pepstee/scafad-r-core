# layers/ migration snapshot (archival)

**Status:** Non-canonical. Archived from working tree on 2026-04-21.

This directory is a frozen copy of the `layers/` intermediate migration
structure that existed between the flat-root era and the current `scafad/`
package architecture. It was never part of any committed HEAD tree after
DL-038 (`ed83f81`), but persisted as an NTFS working-tree artefact.

## Why it exists

During Phase 1 (DL-019..DL-024), code was reorganised from a flat root
into a `layers/` hierarchy as a transitional step. The 7-day sprint
(Phase 3, DL-034..DL-038) then promoted all live code into the canonical
`scafad/` package and deleted the `layers/` stubs from HEAD.

## What is canonical

- **`scafad/`** — the only authoritative runtime surface (L0–L6 + runtime)
- **`legacy/root-flat-files/`** — flat-root era, archival only
- **`legacy/layers-migration-snapshot/`** — this directory, archival only

Do not import from this directory. Do not run tests against it.
The `graphify-out/` subdirectory contains a Graphify analysis of the
`layers/` design (not the live codebase — see `GRAPH_REPORT.md`).
