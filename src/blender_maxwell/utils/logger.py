import logging
from pathlib import Path

import rich.console
import rich.logging
import rich.traceback

from blender_maxwell import contracts as ct

from ..nodeps.utils import simple_logger
from ..nodeps.utils.simple_logger import (
	LOG_LEVEL_MAP,  # noqa: F401
	LogLevel,
	loggers,  # noqa: F401
	simple_loggers,  # noqa: F401
	update_all_loggers,  # noqa: F401
	update_logger,  # noqa: F401
)

OUTPUT_CONSOLE = rich.console.Console(
	color_system='truecolor',
	## TODO: color_system should be 'auto'; bl_run.py hijinks are interfering
)
ERROR_CONSOLE = rich.console.Console(
	color_system='truecolor',
	stderr=True,
	## TODO: color_system should be 'auto'; bl_run.py hijinks are interfering
)
rich.traceback.install(show_locals=True, console=ERROR_CONSOLE)


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
		console=ERROR_CONSOLE,
		rich_tracebacks=True,
	)
	rich_handler.setFormatter(rich_formatter)
	return rich_handler


def file_handler(path_log_file: Path, level: LogLevel) -> rich.logging.RichHandler:
	return simple_logger.file_handler(path_log_file, level)


####################
# - Logger Setup
####################
def get(module_name):
	logger = logging.getLogger(module_name)

	# Setup Logger from Addon Preferences
	ct.addon.prefs().on_addon_logging_changed(single_logger_to_setup=logger)

	return logger
