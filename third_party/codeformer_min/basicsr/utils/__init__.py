import logging


def get_root_logger(logger_name='basicsr', log_level=logging.INFO,
                    log_file=None):
    """最小替代: 官方 basicsr.utils.get_root_logger"""
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(h)
    logger.setLevel(log_level)
    return logger
