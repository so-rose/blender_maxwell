import contextlib
import importlib.metadata
import os
import sys
from pathlib import Path

from ... import info
from . import simple_logger

log = simple_logger.get(__name__)

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

	log.info('Adding Path to sys.path: %s', str(os_path))
	sys.path.insert(0, os_path)
	try:
		yield
	finally:
		log.info('Removing Path from sys.path: %s', str(os_path))
		sys.path.remove(os_path)


@contextlib.contextmanager
def syspath_from_bpy_prefs() -> bool:
	import bpy

	addon_prefs = bpy.context.preferences.addons[info.ADDON_NAME].preferences
	if hasattr(addon_prefs, 'path_addon_pydeps'):
		log.info('Retrieved PyDeps Path from Addon Prefs')
		path_pydeps = addon_prefs.path_addon_pydeps
		with importable_addon_deps(path_pydeps):
			yield True
	else:
		log.info("Couldn't PyDeps Path from Addon Prefs")
		yield False


####################
# - Check PyDeps
####################
def _check_pydeps(
	path_requirementslock: Path,
	path_deps: Path,
) -> dict[str, tuple[str, str]]:
	"""Check if packages defined in a 'requirements.lock' file are currently installed.

	Returns a list of any issues (if empty, then all dependencies are correctly satisfied).
	"""

	def conform_pypi_package_deplock(deplock: str):
		"""Conforms a <package>==<version> de-lock to match if pypi considers them the same (PyPi is case-insensitive and considers -/_ to be the same)

		See <https://peps.python.org/pep-0426/#name>
		"""
		return deplock.lower().replace('_', '-')

	with path_requirementslock.open('r') as file:
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
	return (
		[
			f'{name}: Have {inst_ver}, Need {req_ver}'
			for name, (req_ver, inst_ver) in conflicts.items()
		]
		+ [
			f'Missing {deplock}'
			for deplock in req_not_inst
			if deplock.split('==')[0] not in conflicts
		]
		+ [
			f'Superfluous {deplock}'
			for deplock in inst_not_req
			if deplock.split('==')[0] not in conflicts
		]
	)


####################
# - Refresh PyDeps
####################
def check_pydeps(path_deps: Path):
	global DEPS_OK  # noqa: PLW0603
	global DEPS_ISSUES  # noqa: PLW0603

	if len(issues := _check_pydeps(info.PATH_REQS, path_deps)) > 0:
		log.info('PyDeps Check Failed')
		log.debug('%s', ', '.join(issues))

		DEPS_OK = False
		DEPS_ISSUES = issues
	else:
		log.info('PyDeps Check Succeeded')
		DEPS_OK = True
		DEPS_ISSUES = []

	return DEPS_OK
