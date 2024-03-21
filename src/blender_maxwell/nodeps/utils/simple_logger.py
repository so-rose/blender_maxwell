import logging
import typing as typ
from pathlib import Path

LogLevel: typ.TypeAlias = int
LogHandler: typ.TypeAlias = typ.Any  ## TODO: Can we do better?

####################
# - Constants
####################
LOG_LEVEL_MAP: dict[str, LogLevel] = {
	'DEBUG': logging.DEBUG,
	'INFO': logging.INFO,
	'WARNING': logging.WARNING,
	'ERROR': logging.ERROR,
	'CRITICAL': logging.CRITICAL,
}

SIMPLE_LOGGER_PREFIX = 'simple::'

STREAM_LOG_FORMAT = 11*' ' + '%(levelname)-8s %(message)s (%(name)s)'
FILE_LOG_FORMAT = STREAM_LOG_FORMAT

####################
# - Globals
####################
CACHE = {
	'console_level': None,
	'file_path': None,
	'file_level': logging.NOTSET,
}


####################
# - Logging Handlers
####################
def console_handler(level: LogLevel) -> logging.StreamHandler:
	stream_formatter = logging.Formatter(STREAM_LOG_FORMAT)
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(stream_formatter)
	stream_handler.setLevel(level)
	return stream_handler


def file_handler(path_log_file: Path, level: LogLevel) -> logging.FileHandler:
	file_formatter = logging.Formatter(FILE_LOG_FORMAT)
	file_handler = logging.FileHandler(path_log_file)
	file_handler.setFormatter(file_formatter)
	file_handler.setLevel(level)
	return file_handler


####################
# - Logger Setup
####################
def setup_logger(
	cb_console_handler: typ.Callable[[LogLevel], LogHandler],
	cb_file_handler: typ.Callable[[Path, LogLevel], LogHandler],
	logger: logging.Logger,
	console_level: LogLevel | None,
	file_path: Path | None,
	file_level: LogLevel,
):
	# Delegate Level Semantics to Log Handlers
	## This lets everything through
	logger.setLevel(logging.DEBUG)

	# DO NOT Propagate to Root Logger
	## This looks like 'double messages'
	logger.propagate = False
	## See SO/6729268/log-messages-appearing-twice-with-python-logging

	# Clear Existing Handlers
	if logger.handlers:
		logger.handlers.clear()

	# Add Console Logging Handler
	if console_level is not None:
		logger.addHandler(cb_console_handler(console_level))

	# Add File Logging Handler
	if file_path is not None:
		logger.addHandler(cb_file_handler(file_path, file_level))


def get(module_name):
	logger = logging.getLogger(SIMPLE_LOGGER_PREFIX + module_name)

	# Reuse Cached Arguments from Last sync_*
	setup_logger(
		console_handler,
		file_handler,
		logger,
		console_level=CACHE['console_level'],
		file_path=CACHE['file_path'],
		file_level=CACHE['file_level'],
	)

	return logger


####################
# - Logger Sync
####################
def sync_bootstrap_logging(
	console_level: LogLevel | None = None,
	file_path: Path | None = None,
	file_level: LogLevel = logging.NOTSET,
):
	CACHE['console_level'] = console_level
	CACHE['file_path'] = file_path
	CACHE['file_level'] = file_level

	logger_logger = logging.getLogger(__name__)
	for name in logging.root.manager.loggerDict:
		logger = logging.getLogger(name)
		setup_logger(
			console_handler,
			file_handler,
			logger,
			console_level=console_level,
			file_path=file_path,
			file_level=file_level,
		)
	logger_logger.info("Bootstrapped Logging w/Settings %s", str(CACHE))


def sync_loggers(
	cb_console_handler: typ.Callable[[LogLevel], LogHandler],
	cb_file_handler: typ.Callable[[Path, LogLevel], LogHandler],
	console_level: LogLevel | None,
	file_path: Path | None,
	file_level: LogLevel,
):
	"""Update all loggers to conform to the given per-handler on/off state and log level."""
	CACHE['console_level'] = console_level
	CACHE['file_path'] = file_path
	CACHE['file_level'] = file_level

	for name in logging.root.manager.loggerDict:
		logger = logging.getLogger(name)
		setup_logger(
			cb_console_handler,
			cb_file_handler,
			logger,
			console_level=console_level,
			file_path=file_path,
			file_level=file_level,
		)


####################
# - Logger Iteration
####################
def loggers():
	return [
		logging.getLogger(name) for name in logging.root.manager.loggerDict
	]


def simple_loggers():
	return [
		logging.getLogger(name)
		for name in logging.root.manager.loggerDict
		if name.startswith(SIMPLE_LOGGER_PREFIX)
	]
