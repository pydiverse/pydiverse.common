# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import datetime as dt
import traceback
import types
from dataclasses import dataclass

import pytest

from pydiverse.common.errors import DisposedError
from pydiverse.common.util import Disposable, deep_map, requires
from pydiverse.common.util.hashing import _hash_polars_dataframe, stable_dataframe_hash

try:
    import polars as pl
except ImportError:
    pl = types.ModuleType("polars")
    pl.DataFrame = None

try:
    import pyarrow as pa
except ImportError:
    pa = types.ModuleType("pyarrow")
    pa.Table = None

try:
    import pandas as pd
except ImportError:
    pd = types.ModuleType("pandas")
    pd.DataFrame = None


def test_requires():
    @requires(None, ImportError("Some Error"))
    class BadClass:
        a = 1
        b = 2

    # Shouldn't be able to create instance
    with pytest.raises(ImportError, match="Some Error"):
        BadClass()

    # Shouldn't be able to access class attribute
    with pytest.raises(ImportError, match="Some Error"):
        _ = BadClass.a

    # If all requirements are fulfilled, nothing should change
    @requires((pytest,), Exception("This shouldn't happen"))
    class GoodClass:
        a = 1

    _ = GoodClass()
    _ = GoodClass.a


def test_disposable():
    class Foo(Disposable):
        a = 1

        def bar(self):
            return 2

    x = Foo()

    assert x.a == 1
    assert x.bar() == 2

    x.dispose()

    with pytest.raises(DisposedError):
        _ = x.a
    with pytest.raises(DisposedError):
        x.foo()
    with pytest.raises(DisposedError):
        x.dispose()
    with pytest.raises(DisposedError):
        x.a = 1


def test_format_exception():
    # traceback.format_exception syntax changed from python 3.9 to 3.10
    # thus we use traceback.format_exc()
    try:
        raise RuntimeError("this error is intended by test")
    except RuntimeError:
        trace = traceback.format_exc()
        assert 'RuntimeError("this error is intended by test")' in trace
        assert "test_util.py" in trace


@dataclass
class Foo:
    a: int
    b: str
    c: list[int]
    d: tuple[int, str]


def test_deep_map():
    assert deep_map(1, lambda n: 2) == 2
    assert deep_map([1], lambda n: 0) == 0
    assert deep_map([1], lambda x: x) == [1]
    assert deep_map([3, 1, None, [3, 1, None, 4], 4], lambda x: 2 if x == 1 else x) == [
        3,
        2,
        None,
        [3, 2, None, 4],
        4,
    ]
    for outer in list, tuple:
        for inner in list, tuple:
            assert deep_map(
                outer([None, 3, 1, inner([3, 1, 4, None]), 4]),
                lambda x: 2 if x == 1 else x,
            ) == outer([None, 3, 2, inner([3, 2, 4, None]), 4])
    # attention: the replacement morphs keys 1 and 2;
    # the latter value overrides but is itself changed from 1 to 2 => 2:2
    assert deep_map({1: 3, 2: 1, 3: None, 4: [3, 1, None, 4], 5: 4}, lambda x: 2 if x == 1 else x) == {
        2: 2,
        3: None,
        4: [3, 2, None, 4],
        5: 4,
    }
    assert deep_map(dict(a=3, b=1, c=None, d=[3, 1, None, 4], e=4), lambda x: 2 if x == 1 else x) == dict(
        a=3, b=2, c=None, d=[3, 2, None, 4], e=4
    )
    assert deep_map(Foo(1, "test", [1, 3], (1, "four")), lambda x: 2 if x == 1 else x) == Foo(
        2, "test", [2, 3], (2, "four")
    )
    assert deep_map([Foo(1, "test", [1, 3], (1, "four"))], lambda x: 2 if x == 1 else x) == [
        Foo(2, "test", [2, 3], (2, "four"))
    ]

    # Currently, deep_map cannot traverse other Iterables than lists, tuples, and dicts.
    d = {1: 1}
    res = deep_map([1, d.values()], lambda x: 2 if x == 1 else x)
    assert res[0] == 2
    assert list(res[1]) == list(d.values())


def check_df_hashes(*dfs, use_polars: bool = True) -> None:
    hashes = []
    for df in dfs:
        hashes.append(stable_dataframe_hash(df))
    # Assert that the hashes are unique
    if use_polars:
        assert len(hashes) == len(set(hashes)), (
            pl.DataFrame(dict(hash=hashes)).with_row_index().group_by("hash").agg(pl.col("index"))
        )
    else:
        assert len(hashes) == len(set(hashes)), (
            pd.DataFrame(dict(hash=hashes)).reset_index().groupby("hash")["index"].apply(list)
        )


def check_df_hashes_polars_specific(*dfs: pl.DataFrame) -> None:
    init_repr_hashes = []
    hashes = []
    for df in dfs:
        init_repr_hashes.append(_hash_polars_dataframe(df, use_init_repr=True))
        hashes.append(_hash_polars_dataframe(df))
        assert _hash_polars_dataframe(df)[0] == "0"
        assert _hash_polars_dataframe(df, use_init_repr=True)[0] == "1"
        assert _hash_polars_dataframe(df) == _hash_polars_dataframe(df)
        assert _hash_polars_dataframe(df, use_init_repr=True) == _hash_polars_dataframe(df, use_init_repr=True)

    # Assert that the hashes are unique
    assert len(hashes) == len(set(hashes)), (
        pl.DataFrame(dict(hash=hashes)).with_row_index().group_by("hash").agg(pl.col("index"))
    )
    assert len(init_repr_hashes) == len(set(init_repr_hashes)), (
        pl.DataFrame(dict(hash=hashes)).with_row_index().group_by("hash").agg(pl.col("index"))
    )


@pytest.mark.skipif(pl.DataFrame is None or pa.Table is None, reason="requires polars and pyarrow")
@pytest.mark.parametrize("use_hash_polars_dataframe", [False, True])
def test_hashing_basic(use_hash_polars_dataframe):
    df_a = pl.DataFrame(dict(x=[1]))
    df_b = pl.DataFrame(dict(y=[1]))
    df_c = pl.DataFrame(dict(x=[2]))
    df_d = pl.DataFrame(dict(x=[1.0]))
    df_e = pl.DataFrame(dict(x=[]))

    if use_hash_polars_dataframe:
        check_df_hashes_polars_specific(df_a, df_b, df_c, df_d, df_e)
    else:
        check_df_hashes(df_a, df_b, df_c, df_d, df_e)


@pytest.mark.skipif(pl.DataFrame is None or pa.Table is None, reason="requires polars and pyarrow")
@pytest.mark.parametrize("use_hash_polars_dataframe", [False, True])
def test_hashing(use_hash_polars_dataframe):
    df_a = pl.DataFrame(data=dict(x=[["foo", "bar"], [""]], y=[[1, 2], None], z=[1, 2])).with_columns(
        s=pl.struct("x", "y")
    )
    df_b = pl.DataFrame(data=dict(x=[["foo", "bar"], [""]], z=[[1, 2], None], y=[1, 2])).with_columns(
        s=pl.struct("x", "y")
    )
    df_c = pl.DataFrame(data=dict(x=[["foo", "bar"], [""]], z=[[1, 2], None], y=[1, 2])).with_columns(
        s=pl.struct("y", "x")
    )
    df_d = pl.DataFrame(data=dict(x=[["foo", "baR"], [""]], y=[[1, 2], None], z=[1, 2])).with_columns(
        s=pl.struct("x", "y")
    )
    df_e = pl.DataFrame(data=dict(x=[["foo", "bar"], [""]], y=[[1, 3], None], z=[1, 2])).with_columns(
        s=pl.struct("x", "y")
    )
    df_f = pl.DataFrame(data=dict(x=[["foo", "bar"], [""]], y=[[1, 3], []], z=[1, 2])).with_columns(
        s=pl.struct("x", "y")
    )
    df_g = pl.DataFrame(data=dict(x=[["foo", "bar"], [""]], y=[[1, 2], None], z=[1, 2])).with_columns(
        s=pl.struct("x", "z")
    )
    df_h = pl.DataFrame(data=dict(x=[["foo", "bar"], [""]], y=[[1, 2], None], z=[1, 2])).with_columns(
        s=pl.struct("x", pl.col("y") * 2)
    )
    df_i = pl.DataFrame(
        data=dict(x=[[{"a": ["foo"]}, {"b": ["bar"]}], [], [{}]], y=[[1, 2], None, []], z=[1, 2, 3])
    ).with_columns(s=pl.struct("x", pl.col("y") * 2))

    df_j = pl.DataFrame({f"a{i}": [1, 2, 3] for i in range(10000)})
    df_j_mod = pl.DataFrame({f"{'b' if i == 5000 else 'a'}{i}": [1, 2, 3] for i in range(10000)})

    if use_hash_polars_dataframe:
        check_df_hashes_polars_specific(df_a, df_b, df_c, df_d, df_e, df_f, df_g, df_h, df_i, df_j, df_j_mod)
    else:
        check_df_hashes(df_a, df_b, df_c, df_d, df_e, df_f, df_g, df_h, df_i, df_j, df_j_mod)


@pytest.mark.skipif(pl.DataFrame is None or pa.Table is None, reason="requires polars and pyarrow")
@pytest.mark.parametrize("use_hash_polars_dataframe", [False, True])
def test_hashing_array(use_hash_polars_dataframe):
    df_a = pl.DataFrame(data=dict(x=[[[1], [2], [3]]]), schema=dict(x=pl.Array(pl.UInt16, shape=(3, 1))))
    df_b = pl.DataFrame(data=dict(y=[[[1], [2], [3]]]), schema=dict(y=pl.Array(pl.UInt16, shape=(3, 1))))
    df_c = pl.DataFrame(data=dict(x=[[[1], [3], [2]]]), schema=dict(x=pl.Array(pl.UInt16, shape=(3, 1))))
    df_d = pl.DataFrame(data=dict(x=[[[1, 2, 3]]]), schema=dict(x=pl.Array(pl.UInt16, shape=(1, 3))))
    df_e = pl.DataFrame(data=dict(x=[[1, 2, 3]]), schema=dict(x=pl.Array(pl.UInt16, shape=3)))

    if use_hash_polars_dataframe:
        check_df_hashes_polars_specific(df_a, df_b, df_c, df_d, df_e)
    else:
        check_df_hashes(df_a, df_b, df_c, df_d, df_e)


@pytest.mark.skipif(pl.DataFrame is None or pa.Table is None, reason="requires polars and pyarrow")
@pytest.mark.parametrize("use_hash_polars_dataframe", [False, True])
def test_hash_categorical_enum(use_hash_polars_dataframe):
    df_a_cat = pl.DataFrame(dict(x=["apple", "banana", "apple"]), schema=dict(x=pl.Categorical))
    df_a_enum = pl.DataFrame(dict(x=["apple", "banana", "apple"]), schema=dict(x=pl.Enum(["apple", "banana"])))
    df_a_str = pl.DataFrame(dict(x=["apple", "banana", "apple"]), schema=dict(x=pl.String))
    df_c_cat = pl.DataFrame(dict(x=["c", "b", "c"]), schema=dict(x=pl.Categorical))
    df_c_enum = pl.DataFrame(dict(x=["c", "b", "c"]), schema=dict(x=pl.Enum(["b", "c"])))
    df_c_enum_2 = pl.DataFrame(dict(x=["c", "b", "c"]), schema=dict(x=pl.Enum(["c", "b"])))
    df_a_cat_2 = pl.DataFrame(dict(x=["apple", "banana", "apple"]), schema=dict(x=pl.Categorical))
    df_d = pl.DataFrame(dict(x=["apple", "apple", "banana"]), schema=dict(x=pl.Categorical))
    df_e = df_a_cat.sort("x")

    if use_hash_polars_dataframe:
        check_df_hashes_polars_specific(df_a_cat, df_a_enum, df_a_str, df_c_cat, df_c_enum, df_c_enum_2, df_d)
        assert _hash_polars_dataframe(df_a_cat) == _hash_polars_dataframe(df_a_cat_2)
        assert _hash_polars_dataframe(df_d) == _hash_polars_dataframe(df_e)
    else:
        check_df_hashes(df_a_cat, df_a_enum, df_a_str, df_c_cat, df_c_enum, df_c_enum_2, df_d)
        assert stable_dataframe_hash(df_a_cat) == stable_dataframe_hash(df_a_cat_2)
        assert stable_dataframe_hash(df_d) == stable_dataframe_hash(df_e)


@pytest.mark.skipif(pd.DataFrame is None or pa.Table is None, reason="requires pandas and pyarrow")
def test_hash_pandas():
    import pandas as pd

    dfs = []

    # Test that the hash changes when we change the backend.
    df_a_arrow = pd.DataFrame({"a": [1, 2, 3]}, dtype="int64[pyarrow]")
    df_a_np = pd.DataFrame({"a": [1, 2, 3]})

    dfs.extend([df_a_arrow, df_a_np])

    # Check that dataframes with many columns are hashed correctly.
    # E.g. repr(df.dtypes) is truncated which can cause changes to be ignored.
    df_b = pd.DataFrame({f"a{i}": [1, 2, 3] for i in range(10000)})
    df_b_mod = pd.DataFrame({f"{'b' if i == 5000 else 'a'}{i}": [1, 2, 3] for i in range(10000)})

    dfs.extend([df_b, df_b_mod])

    # Test that dataframes with datetime objects are hashed correctly.
    df_c_date = pd.DataFrame({"a": [1, 2, 3], "b": [dt.date(2020, 1, 1), dt.date(2021, 2, 2), dt.date(2022, 3, 3)]})
    df_c_datetime = pd.DataFrame(
        {"a": [1, 2, 3], "b": [dt.datetime(2020, 1, 1), dt.datetime(2021, 2, 2), dt.datetime(2022, 3, 3)]}
    )

    dfs.extend([df_c_date, df_c_datetime])

    # Test that dataframes with structs are hashed correctly.
    df_d = pd.DataFrame(
        {"a": [1, 2, 3], "b": [{"a": dt.date(2020, 1, 1)}, {"b": dt.date(2021, 2, 2)}, {"a": dt.date(2022, 3, 3)}]}
    )
    df_d_datetime = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [{"a": dt.datetime(2020, 1, 1, 1)}, {"b": dt.datetime(2021, 2, 2)}, {"a": dt.datetime(2022, 3, 3)}],
        }
    )
    df_d_list = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [{"a": [dt.date(2020, 1, 1)]}, {"b": [dt.date(2021, 2, 2)]}, {"a": [dt.date(2022, 3, 3)]}],
        }
    )
    df_d_empty_list = pd.DataFrame(
        {"a": [1, 2, 3], "b": [{"a": [dt.date(2020, 1, 1)]}, {"b": [dt.date(2021, 2, 2)]}, {"a": []}]}
    )
    df_d_none_list = pd.DataFrame(
        {"a": [1, 2, 3], "b": [{"a": [dt.date(2020, 1, 1)]}, {"b": [dt.date(2021, 2, 2)]}, {"a": [None]}]}
    )
    df_d_none = pd.DataFrame(
        {"a": [1, 2, 3], "b": [{"a": [dt.date(2020, 1, 1)]}, {"b": [dt.date(2021, 2, 2)]}, {"a": None}]}
    )

    dfs.extend([df_d, df_d_datetime, df_d_list, df_d_empty_list, df_d_none_list, df_d_none])

    # Test that dataframes with different indices are hashed differently.
    df_e = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_e_index = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, index=[1, 2, 3])

    dfs.extend([df_e, df_e_index])

    check_df_hashes(*dfs, use_polars=False)

    # Test that dataframes which cannot be represented in pyarrow fail hashing.
    df_f = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", 1]})
    with pytest.raises(pa.ArrowTypeError):
        stable_dataframe_hash(df_f)


@pytest.mark.skipif(pd.DataFrame is None or pa.Table is None, reason="requires pandas and pyarrow")
def test_hash_pandas_datetime_edge_case():
    # This test requires the conversion to CSV of pandas object type columns in the hashing function.
    # due to https://github.com/apache/arrow/issues/41896.
    df_d = pd.DataFrame(
        {"a": [1, 2, 3], "b": [{"a": dt.date(2020, 1, 1)}, {"b": dt.date(2021, 2, 2)}, {"a": dt.date(2022, 3, 3)}]}
    )
    df_d_mixed = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [{"a": dt.datetime(2020, 1, 1, 1)}, {"b": dt.date(2021, 2, 2)}, {"a": dt.date(2022, 3, 3)}],
        }
    )
    check_df_hashes(df_d, df_d_mixed, use_polars=False)


@pytest.mark.skipif(pd.DataFrame is None or pa.Table is None, reason="requires pandas and pyarrow")
def test_hash_pandas_datetime_edge_case_2():
    # This test requires the conversion to CSV of pandas object type columns in the hashing function.
    # due to https://github.com/apache/arrow/issues/41896.
    df_d = pd.DataFrame({"a": [1, 2, 3], "b": [dt.date(2020, 1, 1), dt.date(2021, 2, 2), dt.date(2022, 3, 3)]})
    df_d_mixed = pd.DataFrame(
        {"a": [1, 2, 3], "b": [dt.date(2020, 1, 1), dt.date(2021, 2, 2), dt.datetime(2022, 3, 3, 1)]}
    )
    check_df_hashes(df_d, df_d_mixed, use_polars=False)
