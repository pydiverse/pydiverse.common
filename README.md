# pydiverse.common

[![CI](https://img.shields.io/github/actions/workflow/status/pydiverse/pydiverse.common/tests.yml?style=flat-square&branch=main&label=tests)](https://github.com/pydiverse/pydiverse.common/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/pydiversecommon/badge/?version=latest&style=flat-square)](https://pydiversecommon.readthedocs.io/en/latest)
[![pypi-version](https://img.shields.io/pypi/v/pydiverse-common.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/pydiverse-common)
[![conda-forge](https://img.shields.io/conda/pn/conda-forge/pydiverse-common?logoColor=white&logo=conda-forge&style=flat-square)](https://prefix.dev/channels/conda-forge/packages/pydiverse-common)

A base package for different libraries in the pydiverse library collection.
This includes functionality like a type-system for tabular data (SQL and DataFrame).
This type-system is used for ensuring reliable operation of the pydiverse library
with various execution backends like Pandas, Polars, and various SQL dialects.

## Installation

To install pydiverse common try this:

```bash
git clone https://github.com/pydiverse/pydiverse.common.git
cd pydiverse.common

# Create the environment, activate it and install the pre-commit hooks
pixi install
pixi run pre-commit install
```

## Testing

Tests can be run with:

```bash
pixi run pytest
```

## Packaging and publishing to pypi and conda-forge using github actions

- bump version number in [pyproject.toml](pyproject.toml)
- set correct release date in [changelog.md](docs/source/changelog.md)
- push increased version number to `main` branch
- tag commit with `git tag <version>`, e.g. `git tag 0.7.0`
- `git push --tags`

The package should appear on https://pypi.org/project/pydiverse-common/ in a timely manner. It is normal that it takes
a few hours until the new package version is available on https://conda-forge.org/packages/.

### Packaging and publishing to Pypi manually

Packages are first released on test.pypi.org:

- bump version number in [pyproject.toml](pyproject.toml) (check consistency with [changelog.md](docs/source/changelog.md))
- push increased version number to `main` branch
- `pixi run -e release hatch build`
- `pixi run -e release twine upload --repository testpypi dist/*`
- verify with https://test.pypi.org/search/?q=pydiverse.common

Finally, they are published via:

- `git tag <version>`
- `git push --tags`
- Attention: Please, only continue here, if automatic publishing fails for some reason!
- `pixi run -e release hatch build`
- `pixi run -e release twine upload --repository pypi dist/*`

### Publishing package on conda-forge manually

Conda-forge packages are updated via:

- Attention: Please, only continue here, if automatic conda-forge publishing fails for longer than 24h!
- https://github.com/conda-forge/pydiverse-common-feedstock#updating-pydiverse-common-feedstock
- update `recipe/meta.yaml`
- test meta.yaml in pydiverse common repo: `conda-build build ../pydiverse-common-feedstock/recipe/meta.yaml`
- commit `recipe/meta.yaml` to branch of fork and submit PR
