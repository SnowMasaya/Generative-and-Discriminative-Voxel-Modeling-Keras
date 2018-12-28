#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
# log setting
from logging import getLogger
import logging
import logging.config

# Setting Path
import os.path
from os import path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
APP_ROOT = path.dirname(path.abspath(__file__))


class Logger():
    """
    Logger class
    Referance
        http://docs.python.jp/3/howto/logging.html#logging-advanced-tutorial
    """
    def __init__(self, module):
        """
        setting logger 5 type logger
        we have to use the 5 type log debug, info, warn, error, criticial
        """
        self.module_name = module
        self.conf_file_name = APP_ROOT + '/../config/logging.conf'
        logging.config.fileConfig(self.conf_file_name)
        self.debug_logger = self.__getting_logger("debug")
        self.info_logger = self.__getting_logger("info")
        self.warn_logger = self.__getting_logger("warn")
        self.error_logger = self.__getting_logger("error")
        self.critical_logger = self.__getting_logger("critical")

    def __getting_logger(self, logger_name):
        """
        setting logger paramater
        :param logger_name: we use the 5 type log debug, info, warn,
               error, criticial
        """
        logger = getLogger(logger_name)
        return logger

    def debug_log(self, text):
        """
        call debug log
        :param text: logging text
        """
        self.debug_logger.debug(self.module_name + " - " + text)

    def info_log(self, text):
        """
        call info log
        :param text: logging text
        """
        self.info_logger.info(self.module_name + " - " + text)

    def warn_log(self, text):
        """
        call warn log
        :param text: logging text
        """
        self.warn_logger.warn(self.module_name + " - " + text)

    def error_log(self, text):
        """
        call error log
        :param text: logging text
        """
        self.error_logger.error(self.module_name + " - " + text)

    def critical_log(self, text):
        """
        call critical log
        :param text: logging text
        """
        self.critical_logger.critical(self.module_name + " - " + text)
