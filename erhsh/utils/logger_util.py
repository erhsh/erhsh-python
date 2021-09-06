import logging
import os
import sys


LOG_LEVEL_MAP={
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.INFO,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


def get_logger(level="INFO"):
    # 
    _level = LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
    _formatter = logging.Formatter("%(asctime)s [%(levelname)-7s] [%(process)d] [%(module)s.%(funcName)s():%(lineno)-4d] - %(message)s")

    logger = logging.getLogger()
    logger.setLevel(_level)

    log_file_handler = logging.FileHandler("log.txt")
    log_file_handler.setLevel(_level)
    log_file_handler.setFormatter(_formatter)

    log_console_handler = logging.StreamHandler()
    log_console_handler.setLevel(_level)
    log_console_handler.setFormatter(_formatter)

    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)

    return logger


if __name__ == '__main__':
    logger = get_logger(level="DEBUG")
    logger.debug("this is debug msg")
    logger.info("test param is: %s", "haha")
    logger.warning("this is warning msg")
    logger.error("this is error msg")
