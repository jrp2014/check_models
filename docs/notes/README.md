# Development Notes

This directory contains active reference notes. Historical review, planning,
tooling, and cleanup material is archived in `archive/` or left in git history
once fully resolved.

## Active Documents

- **[GPS_DATA_FORMAT_EXPLANATION.md](GPS_DATA_FORMAT_EXPLANATION.md)** — GPS/EXIF data format reference
- **[UPSTREAM_THINKING_BUDGET_EMITTED_START_ISSUE.md](UPSTREAM_THINKING_BUDGET_EMITTED_START_ISSUE.md)** — Evidence file behind the posted upstream issue [Blaizzy/mlx-vlm#1819](https://github.com/Blaizzy/mlx-vlm/issues/1819) (thinking budget vs. model-emitted `<think>`); kept current with each re-verification against upstream `main` while fix PR #1882 is open

Upstream issue drafts are created under this directory only while a finding is
being prepared or actively tracked; once posted and closed upstream, or
superseded by a newer draft, they move to `archive/`.

## Archive

The `archive/` subdirectory contains historical documents (code reviews, audit
reports, retired tool notes, completed plans and specifications, restructure
plans, migration notes, and resolved backlogs) preserved for reference. See
filenames for dates and topics.

Completed Superpowers plans and designs live under
`archive/superpowers/{plans,specs}/`. There is no active `docs/superpowers/`
tree; start new planning material there only while work is in flight, then move
it into the archive when finished.

## Related Documentation

For current development practices and guidelines, see:

- [../CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [../IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md) - Technical standards
