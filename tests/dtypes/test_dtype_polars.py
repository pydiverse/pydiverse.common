# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.common as pdc
from pydiverse.common import (
    Bool,
    Date,
    Datetime,
    Decimal,
    Dtype,
    Float,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    String,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from pydiverse.common.testing import ALL_TYPES

try:
    import polars as pl
except ImportError:
    pl = None


@pytest.mark.skipif(pl is None, reason="requires polars")
def test_dtype_from_polars():
    def assert_conversion(type_, expected):
        assert Dtype.from_polars(type_) == expected

    assert_conversion(pl.Int64, Int64())
    assert_conversion(pl.Int32, Int32())
    assert_conversion(pl.Int16, Int16())
    assert_conversion(pl.Int8, Int8())

    assert_conversion(pl.UInt64, UInt64())
    assert_conversion(pl.UInt32, UInt32())
    assert_conversion(pl.UInt16, UInt16())
    assert_conversion(pl.UInt8, UInt8())

    assert_conversion(pl.Float64, Float64())
    assert_conversion(pl.Float32, Float32())

    assert_conversion(pl.Utf8, String())
    assert_conversion(pl.Boolean, Bool())

    assert_conversion(pl.Date, Date())
    assert_conversion(pl.Time, Time())
    assert_conversion(pl.Datetime, Datetime())
    assert_conversion(pl.Datetime("ms"), Datetime())
    assert_conversion(pl.Datetime("us"), Datetime())
    assert_conversion(pl.Datetime("ns"), Datetime())


@pytest.mark.skipif(pl is None, reason="requires polars")
def test_dtype_to_polars():
    def assert_conversion(type_: Dtype, expected):
        assert type_.to_polars() == expected

    assert_conversion(Int64(), pl.Int64)
    assert_conversion(Int32(), pl.Int32)
    assert_conversion(Int16(), pl.Int16)
    assert_conversion(Int8(), pl.Int8)

    assert_conversion(UInt64(), pl.UInt64)
    assert_conversion(UInt32(), pl.UInt32)
    assert_conversion(UInt16(), pl.UInt16)
    assert_conversion(UInt8(), pl.UInt8)

    assert_conversion(Float64(), pl.Float64)
    assert_conversion(Float32(), pl.Float32)

    assert_conversion(Decimal(), pl.Decimal(31, 11))
    assert_conversion(Decimal(15), pl.Decimal(15, 6))
    assert_conversion(Decimal(15, 2), pl.Decimal(15, 2))

    assert_conversion(String(), pl.Utf8)
    assert_conversion(String(10), pl.Utf8)
    assert_conversion(Bool(), pl.Boolean)

    assert_conversion(Date(), pl.Date)
    assert_conversion(Time(), pl.Time)
    assert_conversion(Datetime(), pl.Datetime("us"))


@pytest.mark.skipif(pl is None, reason="requires polars")
@pytest.mark.parametrize(
    "type_",
    ALL_TYPES,
)
def test_all_types(type_):
    if type_ is pdc.List:
        type_obj = type_(pdc.Int64())
    elif type_ is pdc.Enum:
        type_obj = type_("a", "bbb", "cc")
    else:
        type_obj = type_()

    acceptance_map = {
        Float: Float64(),
        Int: Int64(),
    }

    dst_type = type_obj.to_polars()
    back_type = Dtype.from_polars(dst_type)
    assert back_type == acceptance_map.get(type_, type_obj)
