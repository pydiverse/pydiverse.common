# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import datetime as dt
import types

import pytest

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
except ImportError:
    np = pa = None
    pd = types.ModuleType("pandas")
    pd.__version__ = "0.0.0"

from pydiverse.common import (
    Bool,
    Date,
    Datetime,
    Dtype,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    PandasBackend,
    String,
    Time,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)


@pytest.mark.skipif(np is None, reason="requires pandas, numpy, and pyarrow")
def test_dtype_from_pandas():
    def assert_conversion(type_, expected):
        assert Dtype.from_pandas(type_) == expected

    assert_conversion(int, Int64())
    assert_conversion(float, Float64())
    assert_conversion(str, String())
    assert_conversion(bool, Bool())

    # Objects should get converted to string
    assert_conversion(object, String())
    assert_conversion(dt.date, String())
    assert_conversion(dt.time, String())
    assert_conversion(dt.datetime, String())

    # Numpy types
    assert_conversion(np.int64, Int64())
    assert_conversion(np.int32, Int32())
    assert_conversion(np.int16, Int16())
    assert_conversion(np.int8, Int8())

    assert_conversion(np.uint64, Uint64())
    assert_conversion(np.uint32, Uint32())
    assert_conversion(np.uint16, Uint16())
    assert_conversion(np.uint8, Uint8())

    assert_conversion(np.floating, Float64())
    assert_conversion(np.float64, Float64())
    assert_conversion(np.float32, Float32())

    assert_conversion(np.bytes_, String())
    assert_conversion(np.bool_, Bool())

    assert_conversion(np.datetime64, Datetime())
    assert_conversion(np.dtype("datetime64[ms]"), Datetime())
    assert_conversion(np.dtype("datetime64[ns]"), Datetime())

    # Numpy nullable extension types
    assert_conversion(pd.Int64Dtype(), Int64())
    assert_conversion(pd.Int32Dtype(), Int32())
    assert_conversion(pd.Int16Dtype(), Int16())
    assert_conversion(pd.Int8Dtype(), Int8())

    assert_conversion(pd.UInt64Dtype(), Uint64())
    assert_conversion(pd.UInt32Dtype(), Uint32())
    assert_conversion(pd.UInt16Dtype(), Uint16())
    assert_conversion(pd.UInt8Dtype(), Uint8())

    assert_conversion(pd.Float64Dtype(), Float64())
    assert_conversion(pd.Float32Dtype(), Float32())

    assert_conversion(pd.StringDtype(), String())
    assert_conversion(pd.BooleanDtype(), Bool())


@pytest.mark.skipif(np is None, reason="requires pandas, numpy, and pyarrow")
def test_dtype_to_pandas_numpy():
    def assert_conversion(type_: Dtype, expected):
        assert type_.to_pandas(PandasBackend.NUMPY) == expected

    assert_conversion(Int64(), pd.Int64Dtype())
    assert_conversion(Int32(), pd.Int32Dtype())
    assert_conversion(Int16(), pd.Int16Dtype())
    assert_conversion(Int8(), pd.Int8Dtype())

    assert_conversion(Uint64(), pd.UInt64Dtype())
    assert_conversion(Uint32(), pd.UInt32Dtype())
    assert_conversion(Uint16(), pd.UInt16Dtype())
    assert_conversion(Uint8(), pd.UInt8Dtype())

    assert_conversion(String(), pd.StringDtype())
    assert_conversion(Bool(), pd.BooleanDtype())

    assert_conversion(Date(), "datetime64[s]")
    assert_conversion(Datetime(), "datetime64[us]")

    with pytest.raises(TypeError):
        Time().to_pandas(PandasBackend.NUMPY)


@pytest.mark.skipif(np is None, reason="requires pandas, numpy, and pyarrow")
@pytest.mark.skipif("pd.__version__ < '2'")
def test_dtype_to_pandas_pyarrow():
    def assert_conversion(type_: Dtype, expected):
        if isinstance(expected, pa.DataType):
            assert type_.to_pandas(PandasBackend.ARROW) == pd.ArrowDtype(expected)
        else:
            assert type_.to_pandas(PandasBackend.ARROW) == expected

    assert_conversion(Int64(), pa.int64())
    assert_conversion(Int32(), pa.int32())
    assert_conversion(Int16(), pa.int16())
    assert_conversion(Int8(), pa.int8())

    assert_conversion(Uint64(), pa.uint64())
    assert_conversion(Uint32(), pa.uint32())
    assert_conversion(Uint16(), pa.uint16())
    assert_conversion(Uint8(), pa.uint8())

    assert_conversion(String(), pd.StringDtype(storage="pyarrow"))
    assert_conversion(Bool(), pa.bool_())

    assert_conversion(Date(), pa.date32())
    assert_conversion(Time(), pa.time64("us"))
    assert_conversion(Datetime(), pa.timestamp("us"))
