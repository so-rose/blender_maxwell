import contextlib
import importlib.metadata
import os
import sys
from pathlib import Path

from . import logger as _logger

log = _logger.get()

####################
# - Constants
####################
PATH_ADDON_ROOT = Path(__file__).resolve().parent.parent
PATH_REQS = PATH_ADDON_ROOT / 'requirements.txt'
DEFAULT_PATH_DEPS = PATH_ADDON_ROOT / '.addon_dependencies'
DEFAULT_PATH_DEPS.mkdir(exist_ok=True)

####################
# - Globals
####################
DEPS_OK: bool | None = None
DEPS_ISSUES: list[str] | None = None


####################
# - sys.path Context Manager
####################
@contextlib.contextmanager
def importable_addon_deps(path_deps: Path):
	os_path = os.fspath(path_deps)
	sys.path.insert(0, os_path)
	try:
		yield
	finally:
		sys.path.remove(os_path)


####################
# - Check PyDeps
####################
def _check_pydeps(
	path_requirementstxt: Path,
	path_deps: Path,
) -> dict[str, tuple[str, str]]:
	"""Check if packages defined in a 'requirements.txt' file are currently installed.

	Returns a list of any issues (if empty, then all dependencies are correctly satisfied).
	"""

	def conform_pypi_package_deplock(deplock: str):
		"""Conforms a <package>==<version> de-lock to match if pypi considers them the same (PyPi is case-insensitive and considers -/_ to be the same)

		See <https://peps.python.org/pep-0426/#name>"""
		return deplock.lower().replace('_', '-')

	with path_requirementstxt.open('r') as file:
		required_depslock = {
			conform_pypi_package_deplock(line)
			for raw_line in file.readlines()
			if (line := raw_line.strip()) and not line.startswith('#')
		}

	# Investigate Issues
	installed_deps = importlib.metadata.distributions(
		path=[str(path_deps.resolve())]  ## resolve() is just-in-case
	)
	installed_depslock = {
		conform_pypi_package_deplock(
			f'{dep.metadata["Name"]}=={dep.metadata["Version"]}'
		)
		for dep in installed_deps
	}

	# Determine Missing/Superfluous/Conflicting
	req_not_inst = required_depslock - installed_depslock
	inst_not_req = installed_depslock - required_depslock
	conflicts = {
		req.split('==')[0]: (req.split('==')[1], inst.split('==')[1])
		for req in req_not_inst
		for inst in inst_not_req
		if req.split('==')[0] == inst.split('==')[0]
	}

	# Assemble and Return Issues
	return [
		f'{name}: Have {inst_ver}, Need {req_ver}'
		for name, (req_ver, inst_ver) in conflicts.items()
	] + [
		f'Missing {deplock}'
		for deplock in req_not_inst
		if deplock.split('==')[0] not in conflicts
	] + [
		f'Superfluous {deplock}'
		for deplock in inst_not_req
		if deplock.split('==')[0] not in conflicts
	]


####################
# - Refresh PyDeps
####################
def check_pydeps(path_deps: Path):
	global DEPS_OK  # noqa: PLW0603
	global DEPS_ISSUES  # noqa: PLW0603

	if len(_issues := _check_pydeps(PATH_REQS, path_deps)) > 0:
		#log.debug('Package Check Failed:', end='\n\t')
		#log.debug(*_issues, sep='\n\t')

		DEPS_OK = False
		DEPS_ISSUES = _issues
	else:
		DEPS_OK = True
		DEPS_ISSUES = _issues

	return DEPS_OK
