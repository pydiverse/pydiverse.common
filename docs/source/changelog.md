# Changelog

## 0.3.12 (2025-08-21)
- fixed Dtype.to_pandas() for parameterized String and Decimal types

## 0.3.11 (2025-08-21)
- fixed __eq__, __hash__, and __repr__ for types with parameters
- import string length from sqlalchemy VARCHAR(n) type

## 0.3.10 (2025-08-21)
- implemented String with max_length argument for SQL VARCHAR(n) generation
- implemented Decimal with precision and scale arguments

## 0.3.9 (2025-08-21)
- fix enum pyarrow dtype

## 0.3.8 (2025-08-19)
- fixed util.hashing.hash_polars_dataframe for simple dataframe

## 0.3.7 (2025-08-19)
- support util.hashing.hash_polars_dataframe

## 0.3.6 (2025-08-01)
- hack structlog / dask / pytest capture incompatibility
(structlog._output.stderr != sys.stderr leads to pickle error)

## 0.3.5 (2025-06-27)
- added enum type

## 0.3.4 (2025-06-10)
- fixed pypi package dependencies

## 0.3.3 (2025-06-08)
- improved support of None and List type
- bug fixes in type conversion functions

## 0.3.2 (2025-06-08)
- pydiverse.common.__version__ (implemented via importlib.metadata)

## 0.3.1 (2025-06-08)
- fixed some type conversions (mostly Duration)

## 0.3.0 (2025-06-08)
- rename Uint type to UInt

## 0.2.1 (2025-06-06)
- also support to_xxx() for generic Int and Float Dtype classes

## 0.2.0 (2025-06-06)
- moved many utility functions from `pydiverse.pipedag` to `pydiverse.common`;
    this includes deep_map, ComputationTracer, @disposable, @requires, stable_hash,
    load_object, and structlog initialization
- Decimal becomes subtype of Float

## 0.1.0 (2022-09-01)
Initial release.

- `@materialize` annotations
- flow definition with nestable stages
- zookeeper synchronization
- postgres database backend
- Prefect 1.x and 2.x support
- multi-processing/multi-node support for state exchange between `@materialize` tasks
- support materialization for: pandas, sqlalchemy, raw sql text, pydiverse.transform
