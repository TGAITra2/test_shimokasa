"""ログ作成用モジュール"""
import logging
import sys

from pythonjsonlogger.jsonlogger import JsonFormatter

fmt = JsonFormatter("%(levelname)%(asctime)%(message)%(filename)%(funcName)%(lineno)")  # type: ignore[no-untyped-call]
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt=fmt)
logging.basicConfig(handlers=[sh], level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Hello")
    logger.warning("Hello")
    logger.error("Hello")
