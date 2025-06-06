# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import logging
import os

import pytest

from pydiverse.common.util.structlog import setup_logging

log_level = logging.INFO if not os.environ.get("DEBUG", "") else logging.DEBUG
setup_logging(log_level=log_level)

try:
    import structlog

    # Pytest Configuration
    @pytest.fixture(autouse=True, scope="function")
    def structlog_test_info(request):
        """Add testcase information to structlog context"""
        if not os.environ.get("DEBUG", ""):
            yield
            return

        with structlog.contextvars.bound_contextvars(testcase=request.node.name):
            yield

    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

logger.info("Logging initialized")
