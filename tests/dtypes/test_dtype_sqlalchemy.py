# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
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

    assert_conversion(Uint64(), sa.BigInteger)
    assert_conversion(Uint32(), sa.BigInteger)
    assert_conversion(Uint16(), sa.Integer)
    assert_conversion(Uint8(), sa.SmallInteger)

    assert_conversion(Float64(), sa.Float)
    assert_conversion(Float32(), sa.Float)

    assert_conversion(String(), sa.String)
    assert_conversion(Bool(), sa.Boolean)

    assert_conversion(Date(), sa.Date)
    assert_conversion(Time(), sa.Time)
    assert_conversion(Datetime(), sa.DateTime)
