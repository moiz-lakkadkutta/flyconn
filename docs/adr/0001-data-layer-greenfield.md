# ADR-0001: Build our own harmonized Parquet/DuckDB data layer (reuse the SJCABS vocabulary, not its files)

Status: Proposed (2026-09-26)

## Context

The brief asks for an offline, pinned, checksummed, harmonized store for MaleCNS
v1.0, FlyWire v630/v783, hemibrain, MANC and BANC. The landscape survey
(`docs/research/landscape_access_analysis.md` §9) found no maintained Python
package doing this. The closest artefacts:

- `sjcabs/fly_connectome_data_tutorial` (MIT): per-dataset Feather/Parquet
  bundles on GCS with a shared metadata vocabulary (`meta_data_entries.csv`), but
  it is a workshop bundle, not a package, ships **MaleCNS v0.9** only, has no
  DuckDB layer, and its helpers assume Google default credentials.
- Official exports are Feather, not Parquet, with three different column
  vocabularies (neuPrint `bodyId_pre/bodyId_post/weight`; FlyWire
  `pre_root_id/post_root_id/syn_count` split per neuropil; BANC `pre/post`).
- `YijieYin/connectome_data_prep` ships `.npz` matrices with unstated dataset
  versions and no license.

## Decision

1. `flyconn.data` owns a YAML registry (name, version, URLs, SHA-256, license,
   citation, schema mapping) and converts official primary files to Parquet
   under a single harmonized schema, queried offline via DuckDB.
2. The harmonized **column vocabulary aligns with the SJCABS/BANC schema**
   (`flow, super_class, cell_class, cell_sub_class, cell_type, side, region,
   neurotransmitter_predicted, <dataset>_<version>_cell_type, *_match_id`) so
   SJCABS, bancr and coconatfly users can round-trip, extended with the columns
   we need (`nt_probs_*`, `status`, `dataset`, `version`).
3. Primary sources are the official bulk files (MaleCNS GCS Feather, FlyWire
   Codex/Zenodo exports, BANC compiled Feather, hemibrain/MANC neuPrint exports),
   not the SJCABS mirrors, so provenance points at the data owners.
4. We contribute a MaleCNS v1.0 bundle and column-mapping table upstream to
   SJCABS once ours is validated.

## Alternatives considered

- Depend on SJCABS files directly: rejected (v0.9, not a package, credentials
  assumption, no checksums, redistribution stance unclear).
- Depend on neuPrint/CAVE live queries: rejected as the default path (tokens,
  online, slow for whole-CNS adjacency; neuPrint tokens issued before 2026-08 are
  now invalid). Kept as optional extras.
- Keep Feather instead of converting to Parquet: rejected; DuckDB reads Parquet
  natively with predicate pushdown, Feather needs the Arrow extension and has no
  row-group statistics.

## Consequences

- One-time conversion cost per dataset (MaleCNS weights: 1.05 GB Feather →
  ~3.6 GB in memory as int64; we will downcast weight to int32 and store sorted
  by `pre` for pushdown). Must stay under 16 GB: convert with pyarrow batch
  iteration, never `to_pandas()` on the full table.
- Unit tests use a committed synthetic fixture plus a tiny real subgraph with
  attribution; no network.

## Evidence

`docs/research/landscape_access_analysis.md` §9–11; `docs/research/data_malecns.md` §4;
`docs/DATA_SOURCES.md`.
