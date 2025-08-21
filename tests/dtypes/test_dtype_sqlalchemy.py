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
    Enum,
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
    import sqlalchemy as sa
except ImportError:
    sa = None


@pytest.mark.skipif(sa is None, reason="requires sqlalchemy")
def test_dtype_from_sqlalchemy():
    def assert_conversion(type_, expected):
        assert Dtype.from_sql(type_) == expected

    assert_conversion(sa.BigInteger(), Int64())
    assert_conversion(sa.Integer(), Int32())
    assert_conversion(sa.SmallInteger(), Int16())

    assert_conversion(sa.Numeric(), Float64())
    assert_conversion(sa.Numeric(13, 2), Float64())
    assert_conversion(sa.Numeric(1, 0), Float64())
    assert_conversion(sa.DECIMAL(13, 2), Float64())
    assert_conversion(sa.DECIMAL(1, 0), Float64())
    assert_conversion(sa.Float(), Float64())
    assert_conversion(sa.Float(24), Float32())
    assert_conversion(sa.Float(53), Float64())

    assert_conversion(sa.String(), String())
    assert_conversion(sa.String(10), String(10))
    assert_conversion(sa.Boolean(), Bool())

    assert_conversion(sa.Date(), Date())
    assert_conversion(sa.Time(), Time())
    assert_conversion(sa.DateTime(), Datetime())


@pytest.mark.skipif(sa is None, reason="requires sqlalchemy")
def test_dtype_to_sqlalchemy():
    def assert_conversion(type_: Dtype, expected):
        assert isinstance(type_.to_sql(), expected)

    assert_conversion(Int64(), sa.BigInteger)
    assert_conversion(Int32(), sa.Integer)
    assert_conversion(Int16(), sa.SmallInteger)
    assert_conversion(Int8(), sa.SmallInteger)

    assert_conversion(UInt64(), sa.BigInteger)
    assert_conversion(UInt32(), sa.BigInteger)
    assert_conversion(UInt16(), sa.Integer)
    assert_conversion(UInt8(), sa.SmallInteger)

    assert_conversion(Float64(), sa.Float)
    assert_conversion(Float32(), sa.Float)

    assert_conversion(Decimal(), sa.Numeric)
    assert_conversion(Decimal(15), sa.Numeric)
    assert_conversion(Decimal(15, 2), sa.Numeric)
    assert Decimal().to_sql().precision == 31
    assert Decimal().to_sql().scale == 11
    assert Decimal(15).to_sql().precision == 15
    assert Decimal(15).to_sql().scale == 6
    assert Decimal(7, 2).to_sql().precision == 7
    assert Decimal(7, 2).to_sql().scale == 2

    assert_conversion(String(), sa.String)
    assert_conversion(String(10), sa.String)
    assert_conversion(Enum("a", "bbb", "cc"), sa.String)
    assert String().to_sql().length is None
    assert String(10).to_sql().length == 10
    assert Enum("a", "bbb", "cc").to_sql().length == 3

    assert_conversion(Bool(), sa.Boolean)

    assert_conversion(Date(), sa.Date)
    assert_conversion(Time(), sa.Time)
    assert_conversion(Datetime(), sa.DateTime)


@pytest.mark.skipif(sa is None, reason="requires sqlalchemy")
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
    dst_type = type_obj.to_sql()
    back_type = Dtype.from_sql(dst_type)
    acceptance_map = {
        # SQL is a bit less strict about integer precisions
        Int8: Int16(),
        UInt8: Int16(),
        UInt16: Int32(),
        UInt32: Int64(),
        UInt64: Int64(),
        # we intentionally fetch Decimal as Float since Decimal is more a relational
        # database thing
        Decimal: Float64(),
        Float: Float64(),
        Int: Int64(),
        # there is no Enum
        Enum: String(3),
    }
    assert back_type == acceptance_map.get(type_, type_obj)
