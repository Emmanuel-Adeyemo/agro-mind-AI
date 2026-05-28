import logging
from pathlib import Path

# ROOT = Path(__file__).resolve().parent.parent
# get logger, check if it has handlers, if not, set format and level, then add handler and save
def logging_setup(name='QGenAI'):

    logger = logging.getLogger()

    # incase handler has already been added to logger
    if logger.hasHandlers():
        return logger

    log_format = logging.Formatter(
        fmt=('%(asctime)s | %(levelname)-8s | %(filename)s: %(lineno)s: %(message)s'),
        datefmt=('%Y-%m-%d %H:%M:%S')
    )

    logger.setLevel(logging.INFO)

    # logging for streamlit
    st_handler = logging.StreamHandler()
    st_handler.setLevel(logging.INFO)
    st_handler.setFormatter(log_format)
    logger.addHandler(st_handler)

    # log out
    # tmp = ROOT / 'tmp'
    log_dir = Path('../../tmp/qgen_log')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'qgen.log'

    # save log
    try:
        file_handler = logging.FileHandler(log_path, encoding='utc-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        # if log writing fails
        logger.warning(f'Could not write log into file due to permission issues: {e}')

    return logger

logger = logging_setup()


