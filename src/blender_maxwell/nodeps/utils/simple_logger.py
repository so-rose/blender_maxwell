import logging
import typing as typ
from pathlib import Path

## TODO: Hygiene; don't try to own all root loggers.

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

STREAM_LOG_FORMAT = 11 * ' ' + '%(levelname)-8s %(message)s (%(name)s)'
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
	"""A logging handler that prints messages to the console.

	Parameters:
		level: The log levels (debug, info, etc.) to print.

	Returns:
		The logging handler, which can be added to a logger.
	"""
	stream_formatter = logging.Formatter(STREAM_LOG_FORMAT)
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(stream_formatter)
	stream_handler.setLevel(level)
	return stream_handler


def file_handler(path_log_file: Path, level: LogLevel) -> logging.FileHandler:
	"""A logging handler that prints messages to a file.

	Parameters:
		path_log_file: The path to the log file.
		level: The log levels (debug, info, etc.) to append to the file.

	Returns:
		The logging handler, which can be added to a logger.
	"""
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
) -> None:
	"""Configures a single logger with given console and file handlers, individualizing the log level that triggers each.

	This is a lower-level function - generally, modules that want to use a well-configured logger will use the `get()` function, which retrieves the parameters for this function from the addon preferences.
	This function is also used by the higher-level log setup.

	Parameters:
		cb_console_handler: A function that takes a log level threshold (inclusive), and returns a logging handler to a console-printer.
		cb_file_handler: A function that takes a log level threshold (inclusive), and returns a logging handler to a file-printer.
		logger: The logger to configure.
		console_level: The log level threshold to print to the console.
			None deactivates file logging.
		path_log_file: The path to the log file.
			None deactivates file logging.
		file_level: The log level threshold to print to the log file.
	"""
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


def get(module_name) -> logging.Logger:
	"""Get a simple logger from the module name.

	Should be used by calling ex. `LOG = simple_logger.get(__name__)` in the module wherein logging is desired.
	Should **only** be used if the dependencies aren't yet available for using `blender_maxwell.utils.logger`.

	Uses the global `CACHE` to store `console_level`, `file_path`, and `file_level`, since addon preferences aren't yet available.

	Parameters:
		module_name: The name of the module to create a logger for.
			Should be set to `__name__`.
	"""
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
) -> None:
	"""Initialize the simple logger, including the `CACHE`, so that logging will work without dependencies / the addon preferences being started yet.

	Should only be called by the addon's pre-initialization code, before `register()`.

	Parameters:
		console_level: The console log level threshold to store in `CACHE`.
			`None` deactivates console logging.
		file_path: The file path to use for file logging, stored in `CACHE`.
			`None` deactivates file logging.
		file_level: The file log level threshold to store in `CACHE`.
			Only needs to be set if `file_path` is not `None`.
	"""
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
	logger_logger.info('Bootstrapped Simple Logging w/Settings %s', str(CACHE))


def sync_all_loggers(
	cb_console_handler: typ.Callable[[LogLevel], LogHandler],
	cb_file_handler: typ.Callable[[Path, LogLevel], LogHandler],
	console_level: LogLevel | None,
	file_path: Path | None,
	file_level: LogLevel,
):
	"""Update all loggers to conform to the given per-handler on/off state and log level.

	This runs the corresponding `setup_logger()` for all active loggers.
	Thus, all parameters are identical to `setup_logger()`.
	"""
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
	return [logging.getLogger(name) for name in logging.root.manager.loggerDict]


def simple_loggers():
	return [
		logging.getLogger(name)
		for name in logging.root.manager.loggerDict
		if name.startswith(SIMPLE_LOGGER_PREFIX)
	]
