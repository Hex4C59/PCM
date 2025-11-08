import logging
import os

def set_logger(log_path: str, level: int = logging.INFO) -> logging.Logger:

    # 确保日志目录存在
    log_dir = os.path.dirname(log_path) if log_path else ""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    # 清理已存在的 handlers，避免重复日志
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # 控制台输出
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    # 文件输出
    file_handler = logging.FileHandler(log_path) if log_path else None
    if file_handler is not None:
        file_handler.setLevel(level)

    # 统一格式（中文简要说明）
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(formatter)
    if file_handler is not None:
        file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    if file_handler is not None:
        logger.addHandler(file_handler)

    return logger
