# pydiverse.common

[![CI](https://github.com/pydiverse/pydiverse.common/actions/workflows/tests.yml/badge.svg)](https://github.com/pydiverse/pydiverse.common/actions/workflows/tests.yml)

A base package for different libraries in the pydiverse library collection.
This includes functionality like a type-system for tabular data (SQL and DataFrame).
This type-system is used for ensuring reliable operation of the pydiverse library
with various execution backends like Pandas, Polars, and various SQL dialects.

## Usage

pydiverse.common can either be installed via pypi with `pip install pydiverse-common` or via
conda-forge with `conda install pydiverse-common -c conda-forge`. Our recommendation would be
to use [pixi](https://pixi.sh/latest/) which is also based on conda-forge:

```bash
mkdir my_project
pixi init
pixi add pydiverse-common
```

With pixi, you run python like this:

```bash
pixi run python -c 'import pydiverse.common'
```

or this:

```bash
pixi run python my_script.py
```
