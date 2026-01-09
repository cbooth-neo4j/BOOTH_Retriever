"""Logging configuration for BOOTH Retriever."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "booth", log_level: str = "INFO") -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Logger name.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    # Check both handlers and a custom attribute to prevent duplicate setup
    if hasattr(logger, '_booth_configured') and logger._booth_configured:
        return logger
    
    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Prevent propagation to parent loggers to avoid duplicate messages
    logger.propagate = False
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Check if handlers already exist before adding
    has_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    # Console handler (less verbose)
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler (more detailed)
    if not has_file:
        log_file = logs_dir / f"booth_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Mark as configured to prevent duplicate setup
    logger._booth_configured = True
    
    return logger


# Create default logger instance
logger = setup_logger()

