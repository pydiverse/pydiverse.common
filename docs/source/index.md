---
hide-toc: true
---

# pydiverse.common

A base package for different libraries in the pydiverse library collection.
This includes functionality like a type-system for tabular data (SQL and DataFrame).
This type-system is used for ensuring reliable operation of the pydiverse library
with various execution backends like Pandas, Polars, and various SQL dialects.

## The Pydiverse Library Collection

Pydiverse is a collection of libraries for describing data transformations and data processing pipelines.

Pydiverse.pipedag is designed to encapsulate any form of data processing pipeline code, providing immediate benefits.
It simplifies the operation of multiple pipeline instances with varying input data sizes and enhances performance
through automatic caching and cache invalidation.
A key objective is to facilitate the iterative improvement of data pipeline code, task by task, stage by stage.

Pydiverse.transform is designed to provide a single syntax for data transformation code that can be executed reliably on
both in-memory dataframes and SQL databases.
The interoperability of tasks in pipedag allows transform to narrow its scope and concentrate on quality.
The results should be identical across different backends, and good error messages should be raised before sending a
query to a backend if a specific feature is not supported.

We are placing increased emphasis on simplifying unit and integration testing across multiple pipeline instances,
which may warrant a separate library called pydiverse.pipetest.

In line with our goal to develop data pipeline code on small input data pipeline instances,
generating test data from the full input data could be an area worth exploring.
This may lead to the creation of a separate library, pydiverse.testdata.

Check out the Pydiverse libraries on GitHub:

- [pydiverse.pipedag](https://github.com/pydiverse/pydiverse.pipedag/)
- [pydiverse.transform](https://github.com/pydiverse/pydiverse.transform/)

Check out the Pydiverse libraries on Read the Docs:

- [pydiverse.pipedag](https://pydiversepipedag.readthedocs.io/en/latest/)
- [pydiverse.transform](https://pydiversetransform.readthedocs.io/en/latest/)


[//]: # (Contents of the Sidebar)

```{toctree}
:maxdepth: 2
:hidden:

reference/api
```

```{toctree}
:caption: Development
:hidden:

changelog
license
```
