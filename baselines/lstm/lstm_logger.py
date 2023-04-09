"""
Create a logger for the lstm baseline
"""

import logging
import sys


class Logger:
    def __init__(self, ):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.ch)

    def info(self, msg):
        self.ch.setLevel(logging.INFO)
        self.logger.info(msg)

    def error(self, msg):
        self.ch.setLevel(logging.ERROR)
        self.logger.error(msg)
