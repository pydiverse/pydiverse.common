# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import pydiverse.common as pdc
from pydiverse.common.testing import ALL_TYPES

try:
    import pyarrow as pa
except ImportError:
    pa = None

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


@pytest.mark.skipif(pa is None, reason="requires pyarrow")
def test_dtype_from_pyarrow():
    def assert_conversion(type_, expected):
        assert Dtype.from_arrow(type_) == expected

    assert_conversion(pa.int64(), Int64())
    assert_conversion(pa.int32(), Int32())
    assert_conversion(pa.int16(), Int16())
    assert_conversion(pa.int8(), Int8())

    assert_conversion(pa.uint64(), UInt64())
    assert_conversion(pa.uint32(), UInt32())
    assert_conversion(pa.uint16(), UInt16())
    assert_conversion(pa.uint8(), UInt8())

    assert_conversion(pa.float64(), Float64())
    assert_conversion(pa.float32(), Float32())
    assert_conversion(pa.float16(), Float32())

    assert_conversion(pa.string(), String())
    assert_conversion(pa.bool_(), Bool())

    assert_conversion(pa.date32(), Date())
    assert_conversion(pa.date64(), Date())

    assert_conversion(pa.time32("s"), Time())
    assert_conversion(pa.time32("ms"), Time())
    assert_conversion(pa.time64("us"), Time())
    assert_conversion(pa.time64("ns"), Time())

    assert_conversion(pa.timestamp("s"), Datetime())
    assert_conversion(pa.timestamp("ms"), Datetime())
    assert_conversion(pa.timestamp("us"), Datetime())
    assert_conversion(pa.timestamp("ns"), Datetime())


@pytest.mark.skipif(pa is None, reason="requires pyarrow")
def test_dtype_to_pyarrow():
    def assert_conversion(type_: Dtype, expected):
        assert type_.to_arrow() == expected

    assert_conversion(Int64(), pa.int64())
    assert_conversion(Int32(), pa.int32())
    assert_conversion(Int16(), pa.int16())
    assert_conversion(Int8(), pa.int8())

    assert_conversion(UInt64(), pa.uint64())
    assert_conversion(UInt32(), pa.uint32())
    assert_conversion(UInt16(), pa.uint16())
    assert_conversion(UInt8(), pa.uint8())

    assert_conversion(Float64(), pa.float64())
    assert_conversion(Float32(), pa.float32())

    assert_conversion(Decimal(), pa.decimal128(31, 11))
    assert_conversion(Decimal(76), pa.decimal256(76, 76 // 3 + 1))
    assert_conversion(Decimal(15), pa.decimal64(15, 6))
    assert_conversion(Decimal(18, 2), pa.decimal64(18, 2))
    assert_conversion(Decimal(9, 9), pa.decimal32(9, 9))

    assert_conversion(String(), pa.string())
    assert_conversion(String(10), pa.string())
    assert_conversion(Bool(), pa.bool_())

    assert_conversion(Date(), pa.date32())
    assert_conversion(Time(), pa.time64("us"))
    assert_conversion(Datetime(), pa.timestamp("us"))


@pytest.mark.skipif(pa is None, reason="requires pyarrow")
def test_dtype_to_pyarrow_enum():
    import polars as pl

    def fix_field_index_type(field):
        if pa.types.is_dictionary(field.type):
            # somehow the index type can jump around in polars and fixing it to uint32
            # seems reasonable
            return (
                pa.field(
                    field.name,
                    pa.dictionary(
                        pa.uint32(), field.type.value_type, field.type.ordered
                    ),
                )
                .with_nullable(field.nullable)
                .with_metadata(field.metadata)
            )
        else:
            return field

    def fix_index_type(schema):
        return pa.schema([fix_field_index_type(field) for field in schema])

    def assert_conversion(type_: Dtype, expected_dtype):
        df = pl.DataFrame(dict(x=["a"]), schema=dict(x=expected_dtype))
        expected = fix_index_type(df.to_arrow().schema)
        actual = pa.schema([type_.to_arrow_field("x", nullable=True)])
        assert actual == expected
        assert actual.field(0).metadata == expected.field(0).metadata

    assert_conversion(Enum("a", "b;c"), pl.Enum(["a", "b;c"]))


@pytest.mark.skipif(pa is None, reason="requires pyarrow")
@pytest.mark.parametrize(
    "type_",
    ALL_TYPES,
)
def test_all_types(type_):
    if type_ is pdc.List:
        type_obj = type_(pdc.Int64())
    elif type_ is pdc.Enum:
        type_obj = type_("a", "b", "c")
    else:
        type_obj = type_()

    acceptance_map = {
        Enum: String(),
        Float: Float64(),
        Int: Int64(),
    }

    dst_type = type_obj.to_arrow()
    back_type = Dtype.from_arrow(dst_type)

    assert back_type == acceptance_map.get(type_, type_obj)
