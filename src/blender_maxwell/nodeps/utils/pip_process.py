import os
import re
import subprocess
import sys
from pathlib import Path

from . import pydeps, simple_logger

log = simple_logger.get(__name__)

PROCESS: subprocess.Popen | None = None
PROGRESS: float | None = None
PROGRESS_FRAC: tuple[str, str] | None = None


def run(reqs_path: Path, pydeps_path: Path, install_log: Path) -> None:
	global PROCESS  # noqa: PLW0603

	if PROCESS is not None:
		msg = 'A pip process is already loaded'
		raise ValueError(msg)

	# Path to Blender's Bundled Python
	## bpy.app.binary_path_python was deprecated in 2.91.
	## sys.executable points to the correct bundled Python.
	## See <https://developer.blender.org/docs/release_notes/2.91/python_api/>
	cmdline = [
		sys.executable,
		'-m',
		'pip',
		'install',
		'-r',
		str(reqs_path),
		'--target',
		str(pydeps_path),
		'--log',
		str(install_log),
		'--disable-pip-version-check',
	]

	log.debug(
		'pip cmdline: %s',
		' '.join(cmdline),
	)

	PROCESS = subprocess.Popen(
		cmdline,
		env=os.environ.copy() | {'PYTHONUNBUFFERED': '1'},
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)


def is_loaded() -> bool:
	return PROCESS is not None


def is_running() -> bool:
	if PROCESS is None:
		msg = "Tried to check whether a process that doesn't exist is running"
		raise ValueError(msg)

	return PROCESS.poll() is None


def returncode() -> bool:
	if not is_running() and PROCESS is not None:
		return PROCESS.returncode

	msg = "Can't get process return code of running/nonexistant process"
	raise ValueError(msg)


def kill() -> None:
	global PROCESS

	if not is_running():
		msg = "Can't kill process that isn't running"
		raise ValueError(msg)

	PROCESS.kill()


def reset() -> None:
	global PROCESS  # noqa: PLW0603
	global PROGRESS  # noqa: PLW0603
	global PROGRESS_FRAC  # noqa: PLW0603

	PROCESS = None
	PROGRESS = None
	PROGRESS_FRAC = None


RE_COLLECTED_DEPLOCK = re.compile(r'Collecting (\w+==[\w\.]+)')


def update_progress(pip_install_log_path: Path):
	global PROGRESS  # noqa: PLW0603
	global PROGRESS_FRAC  # noqa: PLW0603

	if not pip_install_log_path.is_file():
		msg = "Can't parse progress from non-existant pip-install log"
		raise ValueError(msg)

	# start_time = time.perf_counter()
	with pip_install_log_path.open('r') as f:
		pip_install_log = f.read()
	# print('READ', time.perf_counter() - start_time)

	found_deplocks = set(RE_COLLECTED_DEPLOCK.findall(pip_install_log))
	# print('SETUP', time.perf_counter() - start_time)
	PROGRESS = len(found_deplocks) / len(pydeps.DEPS_REQ_DEPLOCKS)
	PROGRESS_FRAC = (str(len(found_deplocks)), str(len(pydeps.DEPS_REQ_DEPLOCKS)))
	# print('COMPUTED', time.perf_counter() - start_time)
