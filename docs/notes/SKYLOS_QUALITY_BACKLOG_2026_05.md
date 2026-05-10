# Skylos Quality Backlog - May 2026

Skylos quality diagnostics are advisory for now. Gate project cleanup on security,
secret, and dependency findings first, then use these quality results as a
focused refactoring queue.

Keep `src/check_models.py` as the intentional CLI monolith. Refactor only when a
nearby behavior-preserving test path exists, and prefer small readability
improvements over broad rewrites.

## Initial Critical Complexity Queue

Start with the diagnostics most likely to improve maintainability without
changing user-facing behavior:

1. `_build_result_output_cues`
2. `_format_quality_analysis_for_log`
3. `_diagnostics_coverage_and_runtime_section`
4. `_history_transition_detail_text`

Then evaluate the remaining critical hotspots from the first full Skylos scan:

1. `_append_report_markdown_block`
2. `_classify_review_verdict`
3. `_log_compact_metrics`
4. `_log_model_comparison_table_and_charts`
5. `_log_utility_triage`
6. `compare_history_records`

## Working Rules

- Treat full quality count reductions as a secondary outcome, not the immediate
  success metric.
- Add or update tests in existing `src/tests/test_*.py` files before touching
  any behavior that affects report text, diagnostics, history comparison, or
  console logging.
- Keep extraction local to the surrounding report/finalization helpers unless a
  shared abstraction clearly removes repeated logic already present in multiple
  code paths.
- Re-run the targeted test file for the edited behavior plus `make quality`
  before merging each cleanup.
