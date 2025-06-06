# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


class DisposedError(Exception):
    """
    Exception raise when an object has been disposed, but some attributes are
    being accessed nevertheless.
    """
