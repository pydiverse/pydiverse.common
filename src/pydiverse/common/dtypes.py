# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import enum


class PandasBackend(str, enum.Enum):
    NUMPY = "numpy"
    ARROW = "arrow"


class Dtype:
    """Base class for all data types."""

    def __eq__(self, rhs):
        """Return ``True`` if this dtype is equal to `rhs`."""
        return isinstance(rhs, Dtype) and type(self) is type(rhs)

    def __hash__(self):
        """Return a hash for this dtype."""
        return hash(type(self))

    def __repr__(self):
        """Return a string representation of this dtype."""
        return self.__class__.__name__

    @classmethod
    def is_int(cls):
        """Return ``True`` if this dtype is an integer type."""
        return False

    @classmethod
    def is_float(cls):
        """Return ``True`` if this dtype is a float type."""
        return False

    @classmethod
    def is_subtype(cls, rhs):
        """Return ``True`` if this dtype is a subtype of `rhs`.

        For example, ``Int8.is_subtype(Int())`` is ``True``.
        """
        rhs_cls = type(rhs)
        return (
            (cls is rhs_cls)
            or (rhs_cls is Int and cls.is_int())
            or (rhs_cls is Float and cls.is_float())
        )

    @staticmethod
    def from_sql(sql_type) -> "Dtype":
        """Convert a SQL type to a Dtype."""
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
        if isinstance(sql_type, sqa.Interval):
            return Duration()
        if isinstance(sql_type, sqa.ARRAY):
            return List(Dtype.from_sql(sql_type.item_type))
        if isinstance(sql_type, sqa.types.NullType):
            return NullType()

        raise TypeError

    @staticmethod
    def from_pandas(pandas_type) -> "Dtype":
        """Convert a pandas type to a Dtype."""
        import numpy as np
        import pandas as pd

        if isinstance(pandas_type, pd.ArrowDtype):
            return Dtype.from_arrow(pandas_type.pyarrow_dtype)

        def is_np_dtype(type_, np_dtype):
            return pd.core.dtypes.common._is_dtype_type(
                type_, pd.core.dtypes.common.classes(np_dtype)
            )

        workaround = (
            pandas_type is not np.floating
        )  # see https://github.com/pandas-dev/pandas/issues/62018
        if workaround and pd.api.types.is_signed_integer_dtype(pandas_type):
            if is_np_dtype(pandas_type, np.int64):
                return Int64()
            elif is_np_dtype(pandas_type, np.int32):
                return Int32()
            elif is_np_dtype(pandas_type, np.int16):
                return Int16()
            elif is_np_dtype(pandas_type, np.int8):
                return Int8()
            raise TypeError
        if workaround and pd.api.types.is_unsigned_integer_dtype(pandas_type):
            if is_np_dtype(pandas_type, np.uint64):
                return UInt64()
            elif is_np_dtype(pandas_type, np.uint32):
                return UInt32()
            elif is_np_dtype(pandas_type, np.uint16):
                return UInt16()
            elif is_np_dtype(pandas_type, np.uint8):
                return UInt8()
            raise TypeError
        if not workaround or pd.api.types.is_float_dtype(pandas_type):
            if not workaround or is_np_dtype(pandas_type, np.float64):
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
        if pd.api.types.is_timedelta64_dtype(pandas_type):
            return Duration()
        # we don't know any decimal/time/null dtypes in pandas if column is not
        # arrow backed

        if pandas_type.name == "category":
            return Enum(*pandas_type.categories.to_list())

        raise TypeError

    @staticmethod
    def from_arrow(arrow_type) -> "Dtype":
        """Convert a PyArrow type to a Dtype."""
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
                return UInt64()
            if pa.types.is_uint32(arrow_type):
                return UInt32()
            if pa.types.is_uint16(arrow_type):
                return UInt16()
            if pa.types.is_uint8(arrow_type):
                return UInt8()
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
        if pa.types.is_duration(arrow_type):
            return Duration()
        if pa.types.is_null(arrow_type):
            return NullType()
        if pa.types.is_list(arrow_type):
            return List(Dtype.from_arrow(arrow_type.value_type))
        raise TypeError

    @staticmethod
    def from_polars(polars_type) -> "Dtype":
        """Convert a Polars type to a Dtype."""
        import polars as pl

        if isinstance(polars_type, pl.List):
            return List(Dtype.from_polars(polars_type.inner))
        if isinstance(polars_type, pl.Enum):
            return Enum(*polars_type.categories)

        return {
            pl.Int64: Int64(),
            pl.Int32: Int32(),
            pl.Int16: Int16(),
            pl.Int8: Int8(),
            pl.UInt64: UInt64(),
            pl.UInt32: UInt32(),
            pl.UInt16: UInt16(),
            pl.UInt8: UInt8(),
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
        }[polars_type.base_type()]

    def to_sql(self):
        """Convert this Dtype to a SQL type."""
        import sqlalchemy as sqa

        return {
            Int(): sqa.BigInteger(),  # we default to 64 bit
            Int8(): sqa.SmallInteger(),
            Int16(): sqa.SmallInteger(),
            Int32(): sqa.Integer(),
            Int64(): sqa.BigInteger(),
            UInt8(): sqa.SmallInteger(),
            UInt16(): sqa.Integer(),
            UInt32(): sqa.BigInteger(),
            UInt64(): sqa.BigInteger(),
            Float(): sqa.Float(53),  # we default to 64 bit
            Float32(): sqa.Float(24),
            Float64(): sqa.Float(53),
            Decimal(): sqa.DECIMAL(),
            String(): sqa.String(),
            Bool(): sqa.Boolean(),
            Date(): sqa.Date(),
            Time(): sqa.Time(),
            Datetime(): sqa.DateTime(),
            Duration(): sqa.Interval(),
            NullType(): sqa.types.NullType(),
        }[self]

    def to_pandas(self, backend: PandasBackend = PandasBackend.ARROW):
        """Convert this Dtype to a pandas type."""
        import pandas as pd

        if backend == PandasBackend.NUMPY:
            return self.to_pandas_nullable(backend)
        if backend == PandasBackend.ARROW:
            if self == String():
                return pd.StringDtype(storage="pyarrow")
            return pd.ArrowDtype(self.to_arrow())

    def to_pandas_nullable(self, backend: PandasBackend = PandasBackend.ARROW):
        """Convert this Dtype to a pandas nullable type.

        Nullable can be either pandas extension types like StringDtype or ArrowDtype.

        Parameters
        ----------
        backend : PandasBackend, optional
            The pandas backend to use. Defaults to ``PandasBackend.ARROW``.
            If ``PandasBackend.NUMPY`` is selected, this method will attempt
            to return a NumPy-backed nullable pandas dtype. Note that
            Time, NullType, and List will raise a TypeError for the
            NUMPY backend as pandas doesn't have corresponding native
            nullable dtypes for these.
        """
        import pandas as pd

        if backend == PandasBackend.ARROW:
            return pd.ArrowDtype(self.to_arrow())

        # we don't want to produce object columns
        if isinstance(self, Time):
            raise TypeError("pandas doesn't have a native time dtype")
        if isinstance(self, NullType):
            raise TypeError("pandas doesn't have a native null dtype")
        if isinstance(self, List):
            raise TypeError("pandas doesn't have a native list dtype")

        if isinstance(self, Enum):
            return pd.CategoricalDtype(self.categories)

        return {
            Int(): pd.Int64Dtype(),  # we default to 64 bit
            Int8(): pd.Int8Dtype(),
            Int16(): pd.Int16Dtype(),
            Int32(): pd.Int32Dtype(),
            Int64(): pd.Int64Dtype(),
            UInt8(): pd.UInt8Dtype(),
            UInt16(): pd.UInt16Dtype(),
            UInt32(): pd.UInt32Dtype(),
            UInt64(): pd.UInt64Dtype(),
            Float(): pd.Float64Dtype(),  # we default to 64 bit
            Float32(): pd.Float32Dtype(),
            Float64(): pd.Float64Dtype(),
            Decimal(): pd.Float64Dtype(),  # NumericDtype is
            String(): pd.StringDtype(),
            Bool(): pd.BooleanDtype(),
            Date(): "datetime64[s]",
            Datetime(): "datetime64[us]",
            Time(): "timedelta64[us]",
            Duration(): "timedelta64[us]",
        }[self]

    def to_arrow(self):
        """Convert this Dtype to a PyArrow type."""
        import pyarrow as pa

        if isinstance(self, Enum):
            return pa.string()

        return {
            Int(): pa.int64(),  # we default to 64 bit
            Int8(): pa.int8(),
            Int16(): pa.int16(),
            Int32(): pa.int32(),
            Int64(): pa.int64(),
            UInt8(): pa.uint8(),
            UInt16(): pa.uint16(),
            UInt32(): pa.uint32(),
            UInt64(): pa.uint64(),
            Float(): pa.float64(),  # we default to 64 bit
            Float32(): pa.float32(),
            Float64(): pa.float64(),
            Decimal(): pa.decimal128(35, 10),  # Arbitrary precision
            String(): pa.string(),
            Bool(): pa.bool_(),
            Date(): pa.date32(),
            Time(): pa.time64("us"),
            Datetime(): pa.timestamp("us"),
            Duration(): pa.duration("us"),
            NullType(): pa.null(),
        }[self]

    def to_polars(self: "Dtype"):
        """Convert this Dtype to a Polars type."""
        import polars as pl

        return {
            Int(): pl.Int64,  # we default to 64 bit
            Int64(): pl.Int64,
            Int32(): pl.Int32,
            Int16(): pl.Int16,
            Int8(): pl.Int8,
            UInt64(): pl.UInt64,
            UInt32(): pl.UInt32,
            UInt16(): pl.UInt16,
            UInt8(): pl.UInt8,
            Float(): pl.Float64,  # we default to 64 bit
            Float64(): pl.Float64,
            Float32(): pl.Float32,
            Decimal(): pl.Decimal(scale=10),  # Arbitrary precision
            String(): pl.Utf8,
            Bool(): pl.Boolean,
            Datetime(): pl.Datetime("us"),
            Duration(): pl.Duration("us"),
            Time(): pl.Time,  # Polars uses nanoseconds since midnight
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


class UInt64(Int): ...


class UInt32(Int): ...


class UInt16(Int): ...


class UInt8(Int): ...


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

    def to_arrow(self):
        import pyarrow as pa

        return pa.list_(self.inner.to_arrow())


class Enum(String):
    def __init__(self, *categories: str):
        if not all(isinstance(c, str) for c in categories):
            raise TypeError("arguments for `Enum` must have type `str`")
        self.categories = list(categories)

    def __eq__(self, rhs):
        return isinstance(rhs, Enum) and self.categories == rhs.categories

    def __repr__(self) -> str:
        return f"Enum[{', '.join(repr(c) for c in self.categories)}]"

    def __hash__(self):
        return hash(tuple(self.categories))

    def to_polars(self):
        import polars as pl

        return pl.Enum(self.categories)

    def to_sql(self):
        import sqlalchemy as sqa

        return sqa.String()

    def to_arrow(self):
        import pyarrow as pa

        # There is also pa.dictionary(), which seems to be kind of similar to an enum.
        # Maybe it is better to convert to this.
        return pa.string()
