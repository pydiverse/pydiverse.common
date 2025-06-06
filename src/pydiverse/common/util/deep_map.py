# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Generic deep map or mutation operations.

Heavily inspired by the builtin copy module of python:
https://github.com/python/cpython/blob/main/Lib/copy.py
"""

from collections.abc import Callable

from .computation_tracing import fully_qualified_name
from .import_ import load_object

_nil = []


def deep_map(x, fn: Callable, memo=None):
    if memo is None:
        memo = {}

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    cls = type(x)

    if cls == list:  # noqa: E721
        y = _deep_map_list(x, fn, memo)
    elif cls == tuple:  # noqa: E721
        y = _deep_map_tuple(x, fn, memo)
    elif cls == dict:  # noqa: E721
        y = _deep_map_dict(x, fn, memo)
    elif hasattr(cls, "__dataclass_fields__"):
        # reconstruct data classes
        y = load_object(
            {
                "class": fully_qualified_name(cls),
                "args": _deep_map_dict(x.__dict__, fn, memo),
            }
        )
    else:
        y = fn(x)

    # If is its own copy, don't memoize.
    if y is not x:
        memo[d] = y
        _keep_alive(x, memo)  # Make sure x lives at least as long as d

    return y


def _deep_map_list(x, fn, memo):
    y = []
    append = y.append
    for a in x:
        append(deep_map(a, fn, memo))
    return fn(y)


def _deep_map_tuple(x, fn, memo):
    y = [deep_map(a, fn, memo) for a in x]
    # We're not going to put the tuple in the memo, but it's still important we
    # check for it, in case the tuple contains recursive mutable structures.
    try:
        return memo[id(x)]
    except KeyError:
        pass
    for k, j in zip(x, y, strict=False):
        if k is not j:
            y = tuple(y)
            break
    else:
        y = x
    return fn(y)


def _deep_map_dict(x, fn, memo):
    y = {}
    memo[id(x)] = y
    for key, value in x.items():
        y[deep_map(key, fn, memo)] = deep_map(value, fn, memo)
    return fn(y)


def _keep_alive(x, memo):
    """Keeps a reference to the object x in the memo.
    Because we remember objects by their id, we have
    to assure that possibly temporary objects are kept
    alive by referencing them.
    We store a reference at the id of the memo, which should
    normally not be used unless someone tries to deepcopy
    the memo itself...
    """
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # aha, this is the first one :-)
        memo[id(memo)] = [x]
