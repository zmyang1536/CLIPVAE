import logging
import os
from typing import Optional


def setup_logger(
        log_file: Optional[str] = None,
        log_level: int = logging.INFO
) -> None:
    """
    设置日志记录器

    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        log_level: 日志级别
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler()]
    )

    # 如果指定了日志文件，添加文件处理器
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logger: 日志记录器
    """
    return logging.getLogger(name)