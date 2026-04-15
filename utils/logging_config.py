"""
Logging configuration and utilities
"""
import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors"""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(
    log_level: str = 'INFO',
    log_dir: Optional[Path] = None,
    app_name: str = 'atlantean_hrm',
    enable_console: bool = True,
    enable_file: bool = True,
    enable_colors: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        app_name: Application name for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_colors: Enable colored console output
        max_bytes: Maximum bytes per log file
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if enable_colors:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_file = log_dir / f'{app_name}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file (ERROR and above only)
        error_log_file = log_dir / f'{app_name}_errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")
    if enable_file and log_dir:
        logger.info(f"Log files in: {log_dir}")
    
    return root_logger


class LoggerContext:
    """Context manager for temporary log level changes"""
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize logger context
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.level = level
        self.original_level = None
    
    def __enter__(self):
        """Enter context and change log level"""
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore log level"""
        self.logger.setLevel(self.original_level)
        return False


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls
    
    Args:
        logger: Logger to use (defaults to function's module logger)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def log_performance(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function performance
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorated function
    """
    import time
    
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


class StructuredLogger:
    """Logger with structured logging support"""
    
    def __init__(self, name: str):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
    
    def log(self, level: int, message: str, **kwargs):
        """
        Log with structured data
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional structured data
        """
        if kwargs:
            extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
            full_message = f"{message} | {extra_info}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self.log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self.log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self.log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self.log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data"""
        self.log(logging.CRITICAL, message, **kwargs)


def get_logger(name: str) -> logging.Logger:
    """
    Get or create logger with standard configuration
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)
