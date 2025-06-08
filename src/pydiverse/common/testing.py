# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import pydiverse.common.dtypes as dtypes

ALL_TYPES = [
    getattr(dtypes, c)
    for c in dir(dtypes)
    if inspect.isclass(getattr(dtypes, c))
    and issubclass(getattr(dtypes, c), dtypes.Dtype)
    and c != "Dtype"
]
