# Release Process

FleetBench uses a lightweight release process suitable for demos and package testing.

## Versioning

The package version is defined in `pyproject.toml`. Keep the version aligned with:

- `CHANGELOG.md`
- `CITATION.cff`
- Release tag name

## Pre-Release Checklist

```bash
python -m pytest
python -m ruff check src tests scripts examples
python -m ruff format --check src tests scripts examples
python -m mypy
python -m build
python -m twine check dist/*
```

## TestPyPI

The release workflow supports manual TestPyPI publication through GitHub Actions:

1. Open the `Release` workflow.
2. Select `workflow_dispatch`.
3. Choose `testpypi`.
4. Confirm the package page and install from TestPyPI in a clean environment.

## PyPI

Publish to PyPI from a GitHub release or by manually selecting `pypi` in the workflow. The workflow uses trusted publishing and does not require a token in the repository.

## Post-Release

- Confirm the package metadata renders correctly.
- Run a smoke benchmark from the published package.
- Update any demo environments that pin the previous version.
