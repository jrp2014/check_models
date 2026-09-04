# Upstream PR draft: harden GitHub Actions workflows (mlx-vlm)

Paste-ready PR material for `Blaizzy/mlx-vlm`. Prepared from the
`mlx-vlm-idefics3-eos` worktree at `39749dbd` (v0.6.8-76-g39749dbd);
both workflow files are unchanged on upstream `main` at the time of
drafting. Action SHAs below were resolved from the GitHub API on
2026-08-09 (each is the dereferenced commit for the named release tag).
Per the `upstream-mlx-vlm-issues` skill, nothing is filed automatically.

## Suggested branch and commit

```bash
git checkout -b ci/harden-workflow-permissions
git commit -m "ci: pin actions to commit SHAs, restrict token permissions"
```

## PR title

```text
ci: pin actions to commit SHAs and restrict workflow token permissions
```

## PR body

````markdown
### Summary

Hardens the two GitHub Actions workflows against supply-chain drift and
over-broad token grants. No behavior change to tests or publishing.

- Pin every action reference to a full commit SHA (with a `# vX.Y.Z`
  comment so Dependabot/Renovate can still update the pins). Mutable
  tags like `@v4` re-resolve on every run; a compromised or force-moved
  tag would execute unreviewed code with the workflow's token.
- `tests.yml` had no `permissions` block, so jobs received the default
  (potentially write-scoped) `GITHUB_TOKEN`. It now grants read-only
  `contents`, matching what checkout + pytest need.
- `persist-credentials: false` on checkouts: neither workflow pushes,
  so the token doesn't need to stay on disk in the git config.
- The publish job now runs in a `pypi` environment, so the
  `PYPI_API_TOKEN` secret can be scoped to that environment and given
  protection rules instead of being readable by any job in the repo.
  (GitHub creates the environment on first use; moving the secret into
  it is optional but recommended.)
- Bumped the already-SHA-pinned `pypa/gh-action-pypi-publish` from a
  2021-era commit to the current release, and dropped the
  `user`/`packages_dir` inputs that match its defaults.

Flagged by a workflow supply-chain linter while mlx-vlm was checked out
alongside a downstream benchmarking harness; findings verified by hand
against the workflow files.

### Changes

`.github/workflows/tests.yml`:

```yaml
name: Test PRs

on:
  pull_request:
    branches:
      - main

permissions:
  contents: read

jobs:
  test:
    runs-on: macos-14

    steps:
      - name: Checkout code
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false

      - name: Set up Python
        uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: '3.10'

      - name: Install MLX
        run: |
          pip install mlx>=0.15

      - name: Install pre-commit
        run: |
          python -m pip install pre-commit
          pre-commit run --all
          if ! git diff --quiet; then
            echo 'Style checks failed, please install pre-commit and run pre-commit run --all and push the change'
            exit 1
          fi

      - name: Install package and dependencies
        run: |
          python -m pip install pytest
          python -m pip install -e .

      - name: Run Python tests
        run: |
          cd mlx_vlm/
          pytest -s ./tests --ignore=tests/test_smoke.py
```

`.github/workflows/python-publish.yml`:

```yaml
# This workflow will upload a Python Package using Twine when a release is created
# For more information see: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#publishing-to-package-registries

name: Upload Python Package

on:
  release:
    types: [published]

permissions:
  contents: read

jobs:
  deploy:

    runs-on: ubuntu-latest
    environment: pypi

    steps:
    - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      with:
        persist-credentials: false
    - name: Set up Python
      uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build
    - name: Build package
      run: python -m build
    - name: Publish package
      uses: pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33 # v1.14.2
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### Notes for reviewers

- `actions/checkout` v3→v7 and `actions/setup-python` v3/v5→v7 run on
  the node24 runtime, which GitHub-hosted `ubuntu-latest` / `macos-14`
  runners provide; no workflow inputs used here changed semantics.
- `pypa/gh-action-pypi-publish` v1.14.2: `user: __token__` and
  `packages-dir: dist` are the defaults, so those inputs are dropped.
  Publishing behavior with `PYPI_API_TOKEN` is unchanged.
- Possible follow-up (separate PR, needs a PyPI-side setting): switch
  to [trusted publishing](https://docs.pypi.org/trusted-publishers/),
  which replaces the long-lived API token with per-run OIDC
  (`permissions: id-token: write` on the deploy job and no secret at
  all).

### Verification

- `python -c "import yaml, pathlib; [yaml.safe_load(pathlib.Path(p).read_text()) for p in ('.github/workflows/tests.yml', '.github/workflows/python-publish.yml')]"`
  parses both files.
- Each pinned SHA was resolved from the GitHub API as the dereferenced
  commit of the corresponding release tag
  (`actions/checkout@v7.0.1`, `actions/setup-python@v7.0.0`,
  `pypa/gh-action-pypi-publish@v1.14.2`).
- The `tests.yml` job needs only read access: it checks out, installs,
  and runs pytest; nothing writes to the repo or uses the API.
````
