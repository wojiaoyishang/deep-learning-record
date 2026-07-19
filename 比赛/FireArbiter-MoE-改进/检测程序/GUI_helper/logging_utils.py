# -*- coding: utf-8 -*-
"""Console logging and uncaught-exception hooks for the GUI application."""
from __future__ import annotations

import logging
import sys
import threading

LOGGER_NAME = "fire_gui"
logger = logging.getLogger(LOGGER_NAME)


def configure_console_logging() -> logging.Logger:
    """Configure one readable stderr handler without duplicating log lines."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def install_exception_hooks() -> None:
    """Log uncaught exceptions from the GUI thread and Python worker threads."""
    configure_console_logging()

    def _sys_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "未捕获的程序异常",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _sys_hook

    if hasattr(threading, "excepthook"):
        def _thread_hook(args):
            if args.exc_type is KeyboardInterrupt:
                return
            logger.critical(
                "线程中发生未捕获异常：%s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _thread_hook
