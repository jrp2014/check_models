# Completed-Model Summary Tables Design

## Goal

Replace the final pipe-delimited completed-model log with compact, skimmable
tables that remain readable in both the terminal and `check_models.log`.

## Design

Keep the `Completed Models (N)` heading, then group completed models in this
actionability order:

1. Unusable
2. Usable with caveats
3. Usable

The first two groups use a Rich table with `Model`, `Maintainer`, and
`Observations` columns. Observation labels use the existing human-readable,
severity-ordered projection rather than raw codes. The clean `Usable` group uses
a compact single-column model table because its maintainer state and observations
are necessarily `none`.

Tables are rendered through the existing `_log_rich_table()` path so the
terminal and persistent plain-text log receive the same bounded-width layout.
Empty groups are omitted. Model ordering remains deterministic within each group.

During finalization, insert the existing CLI horizontal rule between consecutive
per-model result blocks. Do not add a rule before the first or after the last
model. This keeps each warning and metrics block visibly attached to the summary
above it while reusing the established terminal and log separator style.

A single combined table was rejected because long model and observation values
would reproduce the current wrapping problem. Keeping pipe-delimited bullets was
rejected because Rich cannot align their wrapped continuation lines.

## Testing

Tests exercise the real logging/rendering path and verify that:

- completion groups appear in actionability order;
- observation labels are human-readable and severity-ordered;
- clean completions omit redundant maintainer and observation columns;
- long values wrap within table cells rather than as ambiguous bullet fragments;
- the completed-model count and deterministic model ordering are preserved.
- consecutive per-model result blocks have exactly one separator between the
  first model's metrics and the next model's summary.
