# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import traceback
from dataclasses import dataclass

import pytest

from pydiverse.common.errors import DisposedError
from pydiverse.common.util import Disposable, deep_map, requires


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
    assert deep_map(
        {1: 3, 2: 1, 3: None, 4: [3, 1, None, 4], 5: 4}, lambda x: 2 if x == 1 else x
    ) == {2: 2, 3: None, 4: [3, 2, None, 4], 5: 4}
    assert deep_map(
        dict(a=3, b=1, c=None, d=[3, 1, None, 4], e=4), lambda x: 2 if x == 1 else x
    ) == dict(a=3, b=2, c=None, d=[3, 2, None, 4], e=4)
    assert deep_map(
        Foo(1, "test", [1, 3], (1, "four")), lambda x: 2 if x == 1 else x
    ) == Foo(2, "test", [2, 3], (2, "four"))
    assert deep_map(
        [Foo(1, "test", [1, 3], (1, "four"))], lambda x: 2 if x == 1 else x
    ) == [Foo(2, "test", [2, 3], (2, "four"))]

    # Currently, deep_map cannot traverse other Iterables than lists, tuples, and dicts.
    d = {1: 1}
    res = deep_map([1, d.values()], lambda x: 2 if x == 1 else x)
    assert res[0] == 2
    assert list(res[1]) == list(d.values())
