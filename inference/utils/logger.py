import logging
import sys
import os
from typing import Optional

def setup_logger(
    name: str = "med_cot_eval",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger configuration.
    """
    # 1. Determine if this is the main process (added at the beginning of the function)
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        is_main_process = accelerator.is_main_process
    except ImportError:
        is_main_process = True  # Assume main process if accelerate is not available

    # 2. Main process outputs INFO, other processes only output ERROR
    if not is_main_process:
        level = logging.ERROR

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter to add context information."""

    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        if self.extra:
            context_info = " ".join(f"[{k}={v}]" for k, v in self.extra.items())
            return f"{context_info} {msg}", kwargs
        return msg, kwargs