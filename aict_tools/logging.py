import logging


def setup_logging(logfile=None, verbose=False):

    level = logging.INFO
    if verbose is True:
        level = logging.DEBUG

    log = logging.getLogger()
    log.level = level

    formatter = logging.Formatter(
        fmt='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    log.addHandler(stream)

    if logfile is not None:
        filehandler = logging.FileHandler(logfile)
        filehandler.setFormatter(formatter)

    return log
