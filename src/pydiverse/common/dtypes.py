# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from enum import Enum


class PandasBackend(str, Enum):
    NUMPY = "numpy"
    ARROW = "arrow"


class Dtype:
    """Base class for all data types."""

    def __eq__(self, rhs):
        return isinstance(rhs, Dtype) and type(self) is type(rhs)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def is_int(cls):
        return False

    @classmethod
    def is_float(cls):
        return False

    @classmethod
    def is_subtype(cls, rhs):
        rhs_cls = type(rhs)
        return (
            (cls is rhs_cls)
            or (rhs_cls is Int and cls.is_int())
            or (rhs_cls is Float and cls.is_float())
        )

    @staticmethod
    def from_sql(sql_type) -> "Dtype":
        import sqlalchemy as sqa

        if isinstance(sql_type, sqa.SmallInteger):
            return Int16()
        if isinstance(sql_type, sqa.BigInteger):
            return Int64()
        if isinstance(sql_type, sqa.Integer):
            return Int32()
        if isinstance(sql_type, sqa.Float):
            precision = sql_type.precision or 53
            if precision <= 24:
                return Float32()
            return Float64()
        if isinstance(sql_type, sqa.Numeric | sqa.DECIMAL):
            # Just to be safe, we always use FLOAT64 for fixpoint numbers.
            # Databases are obsessed about fixpoint. However, in dataframes, it
            # is more common to just work with double precision floating point.
            # We see Decimal as subtype of Float. Pydiverse.transform will convert
            # Decimal to Float64 whenever it cannot guarantee semantic correctness
            # otherwise.
            return Float64()
        if isinstance(sql_type, sqa.String):
            return String()
        if isinstance(sql_type, sqa.Boolean):
            return Bool()
        if isinstance(sql_type, sqa.Date):
            return Date()
        if isinstance(sql_type, sqa.Time):
            return Time()
        if isinstance(sql_type, sqa.DateTime):
            return Datetime()
        if isinstance(sql_type, sqa.ARRAY):
            return List(Dtype.from_sql(sql_type.item_type.from_sql))
        if isinstance(sql_type, sqa.Null):
            return NullType()

        raise TypeError

    @staticmethod
    def from_pandas(pandas_type) -> "Dtype":
        import numpy as np
        import pandas as pd

        if isinstance(pandas_type, pd.ArrowDtype):
            return Dtype.from_arrow(pandas_type.pyarrow_dtype)

        def is_np_dtype(type_, np_dtype):
            return pd.core.dtypes.common._is_dtype_type(
                type_, pd.core.dtypes.common.classes(np_dtype)
            )

        if pd.api.types.is_signed_integer_dtype(pandas_type):
            if is_np_dtype(pandas_type, np.int64):
                return Int64()
            elif is_np_dtype(pandas_type, np.int32):
                return Int32()
            elif is_np_dtype(pandas_type, np.int16):
                return Int16()
            elif is_np_dtype(pandas_type, np.int8):
                return Int8()
            raise TypeError
        if pd.api.types.is_unsigned_integer_dtype(pandas_type):
            if is_np_dtype(pandas_type, np.uint64):
                return Uint64()
            elif is_np_dtype(pandas_type, np.uint32):
                return Uint32()
            elif is_np_dtype(pandas_type, np.uint16):
                return Uint16()
            elif is_np_dtype(pandas_type, np.uint8):
                return Uint8()
            raise TypeError
        if pd.api.types.is_float_dtype(pandas_type):
            if is_np_dtype(pandas_type, np.float64):
                return Float64()
            elif is_np_dtype(pandas_type, np.float32):
                return Float32()
            raise TypeError
        if pd.api.types.is_string_dtype(pandas_type):
            # We reserve the use of the object column for string.
            return String()
        if pd.api.types.is_bool_dtype(pandas_type):
            return Bool()
        if pd.api.types.is_datetime64_any_dtype(pandas_type):
            return Datetime()
        # we don't know any decimal dtype in pandas if column is not arrow backed

        raise TypeError

    @staticmethod
    def from_arrow(arrow_type) -> "Dtype":
        import pyarrow as pa

        if pa.types.is_signed_integer(arrow_type):
            if pa.types.is_int64(arrow_type):
                return Int64()
            if pa.types.is_int32(arrow_type):
                return Int32()
            if pa.types.is_int16(arrow_type):
                return Int16()
            if pa.types.is_int8(arrow_type):
                return Int8()
            raise TypeError
        if pa.types.is_unsigned_integer(arrow_type):
            if pa.types.is_uint64(arrow_type):
                return Uint64()
            if pa.types.is_uint32(arrow_type):
                return Uint32()
            if pa.types.is_uint16(arrow_type):
                return Uint16()
            if pa.types.is_uint8(arrow_type):
                return Uint8()
            raise TypeError
        if pa.types.is_floating(arrow_type):
            if pa.types.is_float64(arrow_type):
                return Float64()
            if pa.types.is_float32(arrow_type):
                return Float32()
            if pa.types.is_float16(arrow_type):
                return Float32()
            raise TypeError
        if pa.types.is_decimal(arrow_type):
            # We don't recommend using Decimal in dataframes, but we support it.
            return Decimal()
        if pa.types.is_string(arrow_type):
            return String()
        if pa.types.is_boolean(arrow_type):
            return Bool()
        if pa.types.is_timestamp(arrow_type):
            return Datetime()
        if pa.types.is_date(arrow_type):
            return Date()
        if pa.types.is_time(arrow_type):
            return Time()
        raise TypeError

    @staticmethod
    def from_polars(polars_type) -> "Dtype":
        import polars as pl

        if isinstance(polars_type, pl.List):
            return List(Dtype.from_polars(polars_type.inner))

        return {
            pl.Int64: Int64(),
            pl.Int32: Int32(),
            pl.Int16: Int16(),
            pl.Int8: Int8(),
            pl.UInt64: Uint64(),
            pl.UInt32: Uint32(),
            pl.UInt16: Uint16(),
            pl.UInt8: Uint8(),
            pl.Float64: Float64(),
            pl.Float32: Float32(),
            pl.Decimal: Decimal(),
            pl.Utf8: String(),
            pl.Boolean: Bool(),
            pl.Datetime: Datetime(),
            pl.Time: Time(),
            pl.Date: Date(),
            pl.Null: NullType(),
            pl.Duration: Duration(),
            pl.Enum: String(),
        }[polars_type.base_type()]

    def to_sql(self):
        import sqlalchemy as sqa

        return {
            Int8(): sqa.SmallInteger(),
            Int16(): sqa.SmallInteger(),
            Int32(): sqa.Integer(),
            Int64(): sqa.BigInteger(),
            Uint8(): sqa.SmallInteger(),
            Uint16(): sqa.Integer(),
            Uint32(): sqa.BigInteger(),
            Uint64(): sqa.BigInteger(),
            Float32(): sqa.Float(24),
            Float64(): sqa.Float(53),
            Decimal(): sqa.DECIMAL(),
            String(): sqa.String(),
            Bool(): sqa.Boolean(),
            Date(): sqa.Date(),
            Time(): sqa.Time(),
            Datetime(): sqa.DateTime(),
            NullType(): sqa.types.NullType(),
        }[self]

    def to_pandas(self, backend: PandasBackend = PandasBackend.ARROW):
        import pandas as pd

        if backend == PandasBackend.NUMPY:
            return self.to_pandas_nullable(backend)
        if backend == PandasBackend.ARROW:
            if self == String():
                return pd.StringDtype(storage="pyarrow")
            return pd.ArrowDtype(self.to_arrow())

    def to_pandas_nullable(self, backend: PandasBackend = PandasBackend.ARROW):
        import pandas as pd

        if self == Time():
            if backend == PandasBackend.ARROW:
                return pd.ArrowDtype(self.to_arrow())
            raise TypeError("pandas doesn't have a native time dtype")

        return {
            Int8(): pd.Int8Dtype(),
            Int16(): pd.Int16Dtype(),
            Int32(): pd.Int32Dtype(),
            Int64(): pd.Int64Dtype(),
            Uint8(): pd.UInt8Dtype(),
            Uint16(): pd.UInt16Dtype(),
            Uint32(): pd.UInt32Dtype(),
            Uint64(): pd.UInt64Dtype(),
            Float32(): pd.Float32Dtype(),
            Float64(): pd.Float64Dtype(),
            Decimal(): pd.Float64Dtype(),  # NumericDtype is
            String(): pd.StringDtype(),
            Bool(): pd.BooleanDtype(),
            Date(): "datetime64[s]",
            # Time() not supported
            Datetime(): "datetime64[us]",
        }[self]

    def to_arrow(self):
        import pyarrow as pa

        return {
            Int8(): pa.int8(),
            Int16(): pa.int16(),
            Int32(): pa.int32(),
            Int64(): pa.int64(),
            Uint8(): pa.uint8(),
            Uint16(): pa.uint16(),
            Uint32(): pa.uint32(),
            Uint64(): pa.uint64(),
            Float32(): pa.float32(),
            Float64(): pa.float64(),
            Decimal(): pa.decimal128(35, 10),  # Arbitrary precision
            String(): pa.string(),
            Bool(): pa.bool_(),
            Date(): pa.date32(),
            Time(): pa.time64("us"),
            Datetime(): pa.timestamp("us"),
        }[self]

    def to_polars(self: "Dtype"):
        import polars as pl

        return {
            Int64(): pl.Int64,
            Int32(): pl.Int32,
            Int16(): pl.Int16,
            Int8(): pl.Int8,
            Uint64(): pl.UInt64,
            Uint32(): pl.UInt32,
            Uint16(): pl.UInt16,
            Uint8(): pl.UInt8,
            Float64(): pl.Float64,
            Float32(): pl.Float32,
            Decimal(): pl.Decimal(scale=10),  # Arbitrary precision
            String(): pl.Utf8,
            Bool(): pl.Boolean,
            Datetime(): pl.Datetime("us"),
            Duration(): pl.Duration,
            Time(): pl.Time,
            Date(): pl.Date,
            NullType(): pl.Null,
        }[self]


class Float(Dtype):
    @classmethod
    def is_float(cls):
        return True


class Float64(Float): ...


class Float32(Float): ...


class Decimal(Float): ...


class Int(Dtype):
    @classmethod
    def is_int(cls):
        return True


class Int64(Int): ...


class Int32(Int): ...


class Int16(Int): ...


class Int8(Int): ...


class Uint64(Int): ...


class Uint32(Int): ...


class Uint16(Int): ...


class Uint8(Int): ...


class String(Dtype): ...


class Bool(Dtype): ...


class Datetime(Dtype): ...


class Date(Dtype): ...


class Time(Dtype): ...


class Duration(Dtype): ...


class NullType(Dtype): ...


class List(Dtype):
    def __init__(self, inner: "Dtype"):
        self.inner = inner

    def __eq__(self, rhs):
        return isinstance(rhs, List) and self.inner == rhs.inner

    def __hash__(self):
        return hash((0, hash(self.inner)))

    def __repr__(self):
        return f"List[{repr(self.inner)}]"

    def to_sql(self):
        import sqlalchemy as sqa

        return sqa.ARRAY(self.inner.to_sql())

    def to_polars(self):
        import polars as pl

        return pl.List(self.inner.to_polars())
