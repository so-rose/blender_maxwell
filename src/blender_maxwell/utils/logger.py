import logging
from pathlib import Path

import rich.console
import rich.logging

from .. import info
from ..nodeps.utils import simple_logger
from ..nodeps.utils.simple_logger import (
	LOG_LEVEL_MAP,  # noqa: F401
	LogLevel,
	loggers,  # noqa: F401
	setup_logger,  # noqa: F401
	simple_loggers,  # noqa: F401
	sync_loggers,  # noqa: F401
)


####################
# - Logging Handlers
####################
def console_handler(level: LogLevel) -> rich.logging.RichHandler:
	rich_formatter = logging.Formatter(
		'%(message)s',
		datefmt='[%X]',
	)
	rich_handler = rich.logging.RichHandler(
		level=level,
		console=rich.console.Console(
			color_system='truecolor', stderr=True
		),  ## TODO: Should be 'auto'; bl_run.py hijinks are interfering
		# console=rich.console.Console(stderr=True),
		rich_tracebacks=True,
	)
	rich_handler.setFormatter(rich_formatter)
	return rich_handler


def file_handler(
	path_log_file: Path, level: LogLevel
) -> rich.logging.RichHandler:
	return simple_logger.file_handler(path_log_file, level)


####################
# - Logger Setup
####################
def get(module_name):
	logger = logging.getLogger(module_name)

	# Setup Logger from Addon Preferences
	if (addon_prefs := info.addon_prefs()) is None:
		msg = 'Addon preferences not defined'
		raise RuntimeError(msg)
	addon_prefs.sync_addon_logging(only_sync_logger=logger)

	return logger


####################
# - Logger Sync
####################
#def upgrade_simple_loggers():
#	"""Upgrades simple loggers to rich-enabled loggers."""
#	for logger in simple_loggers():
#		setup_logger(console_handler, file_handler, logger)
