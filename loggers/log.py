import os
import logging

# Setup logging
def setup_logger(log_dir):
    """
    Configure a logger to stream INFO-level messages to stdout and write to a log file.

    Args:
        log_dir (str): Directory path where 'training.log' will be created.

    Returns:
        logging.Logger: Logger named 'train_logger' with console and file handlers.
    """
    # Create or retrieve the named logger
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Create formatter, includes timestamp, log level, and message
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")

    # Stream handler for console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler: writes to 'training.log' in specified directory
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Prevent logs from propagating to root logger
    logger.propagate = False
    return logger
