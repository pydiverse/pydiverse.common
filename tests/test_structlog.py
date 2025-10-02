# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest

from pydiverse.common.util.structlog import capture_logs, log_level

try:
    import structlog
except ImportError:
    structlog = None


@pytest.mark.skipif(structlog is None, reason="requires structlog")
def test_structlog():
    logger = structlog.get_logger(__name__ + ".test_structlog")
    logger.info("start test")
    with structlog.testing.capture_logs() as logs:
        logger.info("This is a test log message", key1="value1", key2=42)
    assert logs == [
        {
            "key1": "value1",
            "key2": 42,
            "event": "This is a test log message",
            "log_level": "info",
        }
    ]


@pytest.mark.skipif(structlog is None, reason="requires structlog")
def test_structlog_level01():
    logger = structlog.get_logger(__name__ + ".test_structlog_level01")
    with capture_logs() as logs:
        with log_level(logging.INFO):
            logger.debug("This is a debug message", key1="value1", key2=42)
            logger.info("This is a test log message", key1="value1", key2=42)
    assert logs == [
        {
            "key1": "value1",
            "key2": 42,
            "event": "This is a test log message",
            "level": "info",
            "log_level": "info",
        }
    ]


@pytest.mark.skipif(structlog is None, reason="requires structlog")
def test_structlog_level02():
    logger = structlog.get_logger(__name__ + ".test_structlog_level02")
    with capture_logs() as logs:
        with log_level(logging.ERROR):
            logger.info("This is a test log message", key1="value1", key2=42)
    assert logs == []
