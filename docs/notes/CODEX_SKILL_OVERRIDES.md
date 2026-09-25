# Repository-scoped Codex skills

[`.codex/config.toml`](../../.codex/config.toml) narrows skill discovery for
this trusted repository without changing personal Codex settings or installed
skill files:

- Superpowers is disabled here. The canonical agent guidance defines the local
  development and validation workflow without a second general process layer.
- The repository's generated `.agents/skills/hf-cli` remains the HF CLI reference.
  Canonical agent guidance selects it instead of loading duplicate personal or
  plugin CLI references. The Hugging Face plugin remains enabled.
- Repository-specific MLX and typing skills, OpenAI's built-in skills, and the
  personal GitHub skills are unchanged.

## Individual skill scope limitation

With Codex CLI `0.158.0-alpha.2`, `skills/list` and `debug prompt-input` still
expose the personal HF CLI skill when its `skills.config` exclusion is set in
the repository config, even though `config/read` reports that setting correctly.
The same exclusion using the exact `SKILL.md` path works as a command-line
override. Directory paths do not work in that check.

Do not retain ineffective exclusions or change personal settings to hide this
repository's duplicates. Until project-scoped skill exclusions are honoured,
the routing rule avoids loading redundant references, but does not remove their
entries from the skill catalog. Recheck this limitation after a Codex update;
do not edit plugin-cache files or disable the whole Hugging Face plugin.

## Scope and reversal

Restart Codex if the current chat still shows the previous skill catalog.
To restore Superpowers here, remove its override from the repository config.
The original skills remain installed and personal defaults still apply outside
this repository. The Codex config does not configure Claude Code; the shared
agent guidance's HF reference selection applies to both agents.

References: [configuration scope](https://learn.chatgpt.com/docs/config-file/config-basic),
[skill enablement](https://learn.chatgpt.com/docs/build-skills#enable-or-disable-local-codex-skills).
