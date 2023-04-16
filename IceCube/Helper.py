import logging
import logging.config
import os
import re
import gc
import psutil


def get_logger(name, msg):
    """
    :param name: string
    :param msg: DEBUG, INFO, WARNING, ERROR
    :return: Logger() instance
    """
    level = {"DEBUG": logging.DEBUG,
             "INFO": logging.INFO,
             "WARNING": logging.WARNING,
             "ERROR": logging.ERROR}
    logging.basicConfig(level=level[msg],
                        format="== %(name)s == %(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger(name)

    return logger


def walk_dir(dirname, batch_ids):
    files = dict()
    pattern = r"_(\d+)\.parquet"

    if batch_ids is None:
        batch_ids = list()
        for base, _, names in os.walk(dirname):
            for name in names:
                match = re.findall(pattern, name)
                batch_ids.append(int(match[0]))
                files[int(match[0])] = os.path.join(base, name)
        return files, batch_ids

    for base, _, names in os.walk(dirname):
        selected_files = dict()
        for name in names:
            match = re.findall(pattern, name)
            if int(match[0]) in batch_ids:
                selected_files[int(match[0])] = os.path.join(base, name)
        files.update(selected_files)
    return files, batch_ids


def memory_check(logger, msg=""):
    gc.collect()
    logger.debug(f"memory usage {psutil.virtual_memory().used / 1024**3:.2f} "
                 f"of {psutil.virtual_memory().total / 1024**3:.2f} GB {msg}")
