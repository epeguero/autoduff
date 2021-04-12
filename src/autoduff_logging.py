import logging
import colorlog

def get_logger():
    # Set up logging with ***colors***.
    _handler = colorlog.StreamHandler()
    _handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(message)s',
        log_colors={
            'DEBUG':    'thin_cyan',
            'INFO':     'thin_white',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        },
    ))

    log = colorlog.getLogger('autoduff')
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    return log

def enable_logger_debug():
    log.setLevel(logging.DEBUG)

log = get_logger()

# class TestSetupError(Exception):
#     """Error setting up test
#     """
