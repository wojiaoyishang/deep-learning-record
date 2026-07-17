# -*- coding: utf-8 -*-
"""FireArbiter GUI entry point.

Run from this directory:
    python GUI.py
"""
from __future__ import annotations

import torch

import sys
from PyQt5 import QtWidgets

from GUI_helper import MainWindow
from GUI_helper.logging_utils import configure_console_logging, install_exception_hooks, logger


def main():
    configure_console_logging()
    install_exception_hooks()
    logger.info("程序启动")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
