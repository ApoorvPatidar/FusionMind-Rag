"""Centralised logging configuration for FusionMind.

All modules should import the pre-configured logger alias::

    from Utils.logger import logging
    logging.info("message")          # works unchanged

For named child loggers (preferred in new code)::

    from Utils.logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""

# DESIGN NOTE: TimedRotatingFileHandler vs a new file per restart
# ---------------------------------------------------------------
# The original code created a brand-new timestamped subdirectory on every
# import, producing one inode-pair per process start.  After ~1 000 restarts
# the logs/ tree accumulates hundreds of directories that confuse log
# aggregators (Datadog, CloudWatch, Splunk) which expect a single stable
# filename to tail.  TimedRotatingFileHandler produces at most
# backupCount+1 = 8 files at any time, rotates atomically at midnight UTC
# (no DST ambiguity), and is natively understood by logrotate and most SIEM
# pipelines.  Disk usage is bounded and grep/awk patterns stay consistent
# across days.

import logging as _logging
import os
import threading
from logging.handlers import TimedRotatingFileHandler

# ---------------------------------------------------------------------------
# Thread-local session context — populated by Flask's before_request hook
# so that every log line emitted during a request automatically carries the
# session_id of the user that triggered it.
# ---------------------------------------------------------------------------
_log_context = threading.local()


def set_log_session_id(session_id: str) -> None:
    """Set the session_id for the current thread's log context."""
    _log_context.session_id = session_id


def clear_log_session_id() -> None:
    """Clear the session_id from the current thread's log context."""
    _log_context.session_id = None


class _SessionFilter(_logging.Filter):
    """Inject session_id from thread-local context into every LogRecord.

    If no session_id has been set for the current thread (e.g. background
    threads, startup code) the field is filled with "-" so the format string
    never raises a KeyError.
    """

    def filter(self, record: _logging.LogRecord) -> bool:
        record.session_id = getattr(_log_context, "session_id", "-") or "-"  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Handler setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, "fusionmind.log")

_fmt = (
    "[ %(asctime)s ] %(name)s session=%(session_id)s "
    "line=%(lineno)d - %(levelname)s - %(message)s"
)
_formatter = _logging.Formatter(_fmt)

_file_handler = TimedRotatingFileHandler(
    LOG_FILE_PATH,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
    utc=True,   # UTC avoids DST-ambiguous log lines during clock changes
)
_file_handler.setFormatter(_formatter)
_file_handler.addFilter(_SessionFilter())

_stream_handler = _logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
_stream_handler.addFilter(_SessionFilter())

# ---------------------------------------------------------------------------
# Root "fusionmind" logger — all named child loggers inherit its handlers
# ---------------------------------------------------------------------------
_root_logger = _logging.getLogger("fusionmind")
_root_logger.setLevel(_logging.INFO)
_root_logger.addHandler(_file_handler)
_root_logger.addHandler(_stream_handler)
_root_logger.propagate = False  # prevent double-emission to the Python root logger


def get_logger(name: str) -> _logging.Logger:
    """Return a named child of the 'fusionmind' root logger.

    All modules should call ``get_logger(__name__)`` instead of
    ``logging.getLogger()`` directly so every log line is routed through
    the shared handler configuration (rotation, stream output,
    session_id filter).
    """
    return _logging.getLogger(f"fusionmind.{name}")


# Re-export the fusionmind root logger as the module-level 'logging' alias.
# All existing ``from Utils.logger import logging`` call sites call
# logging.info() / logging.warning() etc. which are valid Logger methods,
# so they continue to work without any changes.
logging = _root_logger