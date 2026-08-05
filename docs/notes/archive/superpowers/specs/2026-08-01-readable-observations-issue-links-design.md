# Readable Observations and Paste-Safe Issue Links Design

## Goal

Make future human-facing reports understandable when read without the original
prompt, accept conventional Markdown section headings in model output, and keep
cross-file links useful after an issue-ready report is pasted into GitHub.

This change does not rewrite the retained `src/output/` snapshot. The maintainer
will rerun the model matrix after the implementation is merged.

## Catalogue section recognition

The strict catalogue contract still requires non-empty `Title:`, `Description:`,
and `Keywords:` fields. The section parser will additionally accept a standard
one-to-six-hash Markdown heading before those labels, including forms such as:

```markdown
### Title:
Two cats resting on a sofa
```

Existing plain, single-marker, and bold-label forms remain valid. A Markdown
heading does not relax the colon, field-name, non-empty-value, or single-line
title checks. This fixes the observed Pixtral false positives without treating
arbitrary prose headings as a valid structured response.

## Human-readable observations

Stable JSONL observation codes and usability rules remain unchanged. A shared
human presentation layer will translate codes into prompt-independent findings
throughout Markdown, HTML, diagnostics, and gallery output.

Representative labels include:

- `empty_output`: No response text was returned
- `repeated_output`: Response repeats the same text
- `unexpected_special_token`: Unrecognised model control tokens remain visible
- `missing_requested_sections`: Required fields are missing or empty
- `prompt_instruction_echo`: Response repeats the task instructions instead of
  only returning the requested fields
- `unexpected_catalog_preamble`: Extra text appears before the Title field
- `token_cap_truncation`: Response appears cut off at the token limit
- `thinking_trace_incomplete`: Internal reasoning block appears incomplete
- `role_boundary_token_present`: Conversation-role control tokens remain visible
- `thinking_trace_present`: Internal reasoning text remains visible
- `configured_wrapper_present`: Expected model wrapper tokens remain visible
- `minimal_output`: Response is unusually short
- `draft_returned_unchanged`: Title, Description and Keywords copy all supplied
  hints unchanged
- `no_keyword_overlap`: Keywords do not overlap the supplied keyword hints

Where the aggregate issue source retains exact details, the label becomes more
specific. For example, the missing-field observation renders as `Missing or
empty fields: Title, Keywords`.

When a result has several observations, presentation order reflects the
likelihood that the result is unusable rather than detector execution order:

1. no response;
2. repeated text;
3. unrecognised control tokens;
4. missing or empty required fields;
5. repeated task instructions;
6. extra text before Title;
7. apparent token-limit truncation;
8. incomplete reasoning blocks;
9. conversation-role tokens;
10. visible reasoning text;
11. expected wrapper tokens left visible;
12. unusually short output;
13. unchanged supplied hints;
14. no keyword-hint overlap.

This is a display priority only. It does not change assessment or usability
semantics.

## Aggregate issue layout

Actionable crashes remain expanded above all tables with their bounded exception
chain and reproduction command. Other surfaced results are grouped into separate
execution-status sections, emitted only when non-empty:

- Completed attempts requiring review
- Crashed attempts requiring review
- Indeterminate attempts requiring review

Each table uses `Model`, `Usability`, `Observed result`, and `Evidence` columns.
Removing the repeated execution value creates more horizontal space for the
plain-language observation.

## Paste-safe links

The existing `--link-style relative` behavior remains available for local
navigation artifacts such as the output index and gallery. Cross-file links
rendered inside issue-ready artifacts explicitly request canonical repository
URLs under:

`https://github.com/jrp2014/check_models/blob/main/src/output/`

This policy does not depend on mutable global link style. Same-document anchors
remain local (`#diagnostic-...`) because they refer to content included in the
pasted issue body. If an issue-ready artifact later gains another cross-file
link, it must use the same explicit issue-link helper.

## Tests and verification

Tests will be added before production changes and must initially fail for the
missing behavior. They cover:

- exact Pixtral-style `### Title:`, `### Description:`, and `### Keywords:`
  output becoming usable;
- all existing supported label formats remaining valid;
- stable machine observation codes paired with readable, severity-ordered human
  labels;
- separate aggregate tables for completed, crashed, and indeterminate surfaced
  results, without duplicating expanded actionable crashes;
- GitHub-absolute cross-file links in issue-ready output even while the configured
  link style is `relative`;
- continued relative links in local-navigation artifacts;
- generated Markdown lint compliance.

Focused report and quality-analysis tests will run before the prescribed format,
safe Ruff fix, lint, and full `make quality` sequence. Tests and validation use
temporary output directories and do not rewrite `src/output/`.
