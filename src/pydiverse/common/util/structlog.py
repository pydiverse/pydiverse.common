# Copyright (c) QuantCo and pydiverse contributors 2025-2026
# SPDX-License-Identifier: BSD-3-Clause
import logging
import sys
import textwrap
import types
from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO

try:
    import structlog
    from structlog import configure, get_config
    from structlog.testing import LogCapture
    from structlog.typing import EventDict, WrappedLogger

    structlog_installed = True
except ImportError:
    structlog = types.ModuleType("structlog")

    class dev:
        class ConsoleRenderer:
            pass

        plain_traceback = None

    structlog.dev = dev
    structlog_installed = False
    configure, get_config = None, None
    LogCapture = None
    EventDict, WrappedLogger = None, None


class PydiverseConsoleRenderer(structlog.dev.ConsoleRenderer):
    """
    Custom subclass of the structlog ConsoleRenderer that allows rendering
    specific values in the event dict on separate lines.
    """

    def __init__(self, *args, **kwargs):
        self._render_keys = kwargs.pop("render_keys", [])
        super().__init__(*args, **kwargs)

    def __call__(self, logger: WrappedLogger, name: str, event_dict: EventDict):
        render_objects = {}
        for key in self._render_keys:
            obj = event_dict.pop(key, None)
            if obj is not None:
                render_objects[key] = obj

        result = super().__call__(logger, name, event_dict)
        sio = StringIO()
        sio.write(result)

        for key, obj in render_objects.items():
            string_rep = str(obj)
            sio.write(
                "\n"
                + "    ["
                + self._styles.kv_key
                + key
                + self._styles.reset
                + "]"
                + "\n"
                + textwrap.indent(string_rep, prefix="    " + self._styles.kv_value)
                + self._styles.reset
            )

        return sio.getvalue()


def setup_logging(
    log_level=logging.INFO,
    log_stream=None,
    timestamp_format="%Y-%m-%d %H:%M:%S.%f",
    exception_formatter=structlog.dev.plain_traceback,
):
    """Configures structlog and logging with sane defaults."""

    if log_stream is None:
        log_stream = sys.stderr

    # Stdlib: decide where logs go here (handlers), not in structlog’s LoggerFactory
    logging.basicConfig(
        level=log_level,
        handlers=[logging.StreamHandler(log_stream)],
        format="%(message)s",
    )

    if structlog_installed:
        # --- Final renderer used for BOTH stdlib and structlog events ---
        renderer = PydiverseConsoleRenderer(
            render_keys=["query", "table_obj", "task", "table", "detail"], exception_formatter=exception_formatter
        )

        # --- ProcessorFormatter wires stdlib logging -> structlog processors ---
        formatter = structlog.stdlib.ProcessorFormatter(
            # This is the final step for BOTH paths (stdlib & structlog):
            processor=renderer,
            # Processors applied to *foreign* (stdlib logging) records before 'processor'
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.stdlib.add_log_level,
                # structlog.stdlib.filter_by_level,  # for logging, it won't have access to the actual logger
                structlog.processors.TimeStamper(timestamp_format),
            ],
        )

        # --- Stdlib handler (pytest caplog will capture this) ---
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(formatter)

        # It’s often safer to set handlers explicitly than to rely on basicConfig()’s global state
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(log_level)

        # --- structlog side: end with wrap_for_formatter so ProcessorFormatter runs the renderer ---
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.stdlib.add_log_level,
                structlog.stdlib.filter_by_level,  # consult stdlib level
                structlog.processors.TimeStamper(timestamp_format),
                # Hand off to logging's ProcessorFormatter (which will call `renderer`)
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


@contextmanager
def log_level(level: int, logger_name: str | None = None):
    lg = logging.getLogger(logger_name)  # None => root
    old = lg.level
    try:
        lg.setLevel(level)
        yield
    finally:
        lg.setLevel(old)


@contextmanager
def capture_logs() -> Generator[list[EventDict], None, None]:
    """
    Context manager that appends all logging statements to its yielded list
    while it is active. Disables all configured processors for the duration
    of the context manager.

    Attention: this is **not** thread-safe!

    <This is derived from structlog.testing.capture_logs which is MIT licensed>
    """
    assert structlog_installed, "please install structlog to capture logs with this function"

    cap = LogCapture()
    # Modify `_Configuration.default_processors` set via `configure` but always
    # keep the list instance intact to not break references held by bound
    # loggers.
    processors = get_config()["processors"]
    old_processors = processors.copy()
    try:
        # clear processors list and use LogCapture for testing
        processors.clear()
        processors.append(structlog.stdlib.add_log_level)
        processors.append(structlog.stdlib.filter_by_level)
        processors.append(cap)
        configure(processors=processors)
        yield cap.entries
    finally:
        # remove LogCapture and restore original processors
        processors.clear()
        processors.extend(old_processors)
        configure(processors=processors)
