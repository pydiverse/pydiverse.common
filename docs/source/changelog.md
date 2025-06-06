# Changelog

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
