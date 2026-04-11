"""
Logging utilities for SLURM/EXCA job execution.

This module provides utilities to configure Python logging to work properly
with SLURM output files when submitting jobs via EXCA.
"""

import logging
import sys


def configure_logging_for_slurm():
    """
    Configure logging to work properly with SLURM output files.

    This ensures that logger.info(), logger.error(), etc. appear in
    the .out and .err files created by SLURM jobs submitted via EXCA.

    This function should be called at the beginning of any function
    decorated with @infra.apply that runs on SLURM.

    Example:
        @infra.apply
        def my_job_function(self):
            configure_logging_for_slurm()
            logger.info("This will appear in the SLURM .out file")
            logger.error("This will appear in the SLURM .err file")
    """
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler for stdout (INFO and above)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    # Create console handler for stderr (WARNING and above)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stderr_handler.setFormatter(stderr_formatter)

    # Add both handlers to root logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)
    root_logger.setLevel(logging.DEBUG)

    # Also configure EXCA logger for debugging
    logging.getLogger("exca").setLevel(logging.DEBUG)


def configure_logging_for_slurm_all_to_stdout():
    """
    Alternative logging configuration that sends ALL logs to stdout.

    Use this if you prefer to have everything in the .out file instead
    of splitting between .out and .err files.
    """
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler for stdout (ALL levels)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    # Add handler to root logger
    root_logger.addHandler(stdout_handler)
    root_logger.setLevel(logging.DEBUG)

    # Also configure EXCA logger for debugging
    logging.getLogger("exca").setLevel(logging.DEBUG)


def test_logging():
    """Test function to verify logging is working correctly."""
    logger = logging.getLogger(__name__)
    logger.debug("DEBUG: This is a debug message")
    logger.info("INFO: This is an info message")
    logger.warning("WARNING: This is a warning message")
    logger.error("ERROR: This is an error message")
    logger.critical("CRITICAL: This is a critical message")
