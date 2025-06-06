# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from typing import TYPE_CHECKING

import pytest

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
    String,
    Time,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)

pl = pytest.importorskip("polars")

if TYPE_CHECKING:
    import polars as pl


def test_dtype_from_polars():
    def assert_conversion(type_, expected):
        assert Dtype.from_polars(type_) == expected

    assert_conversion(pl.Int64, Int64())
    assert_conversion(pl.Int32, Int32())
    assert_conversion(pl.Int16, Int16())
    assert_conversion(pl.Int8, Int8())

    assert_conversion(pl.UInt64, Uint64())
    assert_conversion(pl.UInt32, Uint32())
    assert_conversion(pl.UInt16, Uint16())
    assert_conversion(pl.UInt8, Uint8())

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


def test_dtype_to_polars():
    def assert_conversion(type_: Dtype, expected):
        assert type_.to_polars() == expected

    assert_conversion(Int64(), pl.Int64)
    assert_conversion(Int32(), pl.Int32)
    assert_conversion(Int16(), pl.Int16)
    assert_conversion(Int8(), pl.Int8)

    assert_conversion(Uint64(), pl.UInt64)
    assert_conversion(Uint32(), pl.UInt32)
    assert_conversion(Uint16(), pl.UInt16)
    assert_conversion(Uint8(), pl.UInt8)

    assert_conversion(Float64(), pl.Float64)
    assert_conversion(Float32(), pl.Float32)

    assert_conversion(String(), pl.Utf8)
    assert_conversion(Bool(), pl.Boolean)

    assert_conversion(Date(), pl.Date)
    assert_conversion(Time(), pl.Time)
    assert_conversion(Datetime(), pl.Datetime("us"))
