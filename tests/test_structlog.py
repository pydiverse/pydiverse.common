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
def test_structlog_level01b():
    logger = logging.getLogger(__name__ + ".test_structlog_level01b")
    logger.setLevel(logging.DEBUG)
    logger.parent.setLevel(logging.INFO)  # be more strict on parent level
    logger.info("update cache")
    with capture_logs() as logs:
        with log_level(logging.DEBUG):
            assert logger.isEnabledFor(logging.DEBUG)
            assert logger.isEnabledFor(logging.INFO)
            logger.info("This is a test log message: %s", dict(key1="value1", key2=42))
            logger.debug("This is a debug message: %s", dict(key1="value1", key2=42))
    assert logs == []  # capture_logs does not work for standard logging


@pytest.mark.skipif(structlog is None, reason="requires structlog")
def test_structlog_level01c():
    logger = structlog.get_logger(__name__ + ".test_structlog_level01")
    logger.setLevel(logging.INFO)
    logger.parent.setLevel(logging.ERROR)  # be more strict on parent level
    logger.info("update cache")
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


@pytest.mark.skipif(structlog is None, reason="requires structlog")
def test_structlog_level03():
    logger = structlog.get_logger(__name__ + ".test_structlog_level02")
    with capture_logs() as logs:
        with log_level(logging.CRITICAL):
            logger.error("This is a test log message", key1="value1", key2=42)
            try:
                raise RuntimeError("this is expected")
            except RuntimeError:
                logger.exception("exception is expected")
    assert logs == []


@pytest.mark.skipif(structlog is None, reason="requires structlog")
def test_structlog_level04():
    logger = structlog.get_logger(__name__ + ".test_structlog_level02")
    with capture_logs() as logs:
        with log_level(logging.ERROR):
            logger.error("This is a test log message", key1="value1", key2=42)
            try:
                raise RuntimeError("this is expected")
            except RuntimeError:
                logger.exception("exception is expected")
    assert logs == [
        {"key1": "value1", "key2": 42, "event": "This is a test log message", "level": "error", "log_level": "error"},
        {"exc_info": True, "event": "exception is expected", "level": "error", "log_level": "error"},
    ]
