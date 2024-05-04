"""Tools for fearless managemenet of addon-specific Python dependencies."""

import contextlib
import importlib.metadata
import os
import sys
from pathlib import Path

import blender_maxwell.contracts as ct

from . import simple_logger

log = simple_logger.get(__name__)

####################
# - Globals
####################
DEPS_OK: bool = False  ## Presume no (but we don't know yet)
DEPS_ISSUES: list[str] = []  ## No known issues (yet)
DEPS_REQ_DEPLOCKS: set[str] = set()
DEPS_INST_DEPLOCKS: set[str] = set()


####################
# - sys.path Context Manager
####################
@contextlib.contextmanager
def importable_addon_deps(path_deps: Path):
	"""Temporarily modifies `sys.path` with a light touch and minimum of side-effects.

	Warnings:
		There are a lot of gotchas with the import system, and this is an enormously imperfect "solution".

	Parameters:
		path_deps:
			Corresponds to the directory into which `pip install --target` was used to install packages.
	"""
	os_path = os.fspath(path_deps)

	if os_path not in sys.path:
		log.info('Adding Path to sys.path: %s', str(os_path))
		sys.path.insert(0, os_path)
		try:
			yield
		finally:
			# TODO: Re-add
			# log.info('Removing Path from sys.path: %s', str(os_path))
			# sys.path.remove(os_path)
			pass
	else:
		try:
			yield
		finally:
			pass


@contextlib.contextmanager
def syspath_from_bpy_prefs() -> bool:
	"""Temporarily modifies `sys.path` using the dependencies found in addon preferences.

	Warnings:
		There are a lot of gotchas with the import system, and this is an enormously imperfect "solution".

	Parameters:
		path_deps: Path to the directory where Python modules can be found.
			Corresponds to the directory into which `pip install --target` was used to install packages.
	"""
	with importable_addon_deps(ct.addon.prefs().pydeps_path):
		log.info('Retrieved PyDeps Path from Addon Prefs')
		yield True


####################
# - Passive PyDeps Checkers
####################
def conform_pypi_package_deplock(deplock: str) -> str:
	"""Conforms a "deplock" string (`<package>==<version>`) so that comparing it with other "deplock" strings will conform to PyPi's matching rules.

	- **Case Sensitivity**: PyPi considers packages with non-matching cases to be the same. _Therefore, we cast all deplocks to lowercase._
	- **Special Characters**: PyPi considers `-` and `_` to be the same character. _Therefore, we replace `_` with `-`_.

	See <https://peps.python.org/pep-0426/#name> for the specification.

	Parameters:
		deplock: The string formatted like `<package>==<version>`.

	Returns:
		The conformed deplock string.
	"""
	return deplock.lower().replace('_', '-')


def compute_required_deplocks(
	path_requirementslock: Path,
) -> set[str]:
	with path_requirementslock.open('r') as file:
		return {
			conform_pypi_package_deplock(line)
			for raw_line in file.readlines()
			if (line := raw_line.strip()) and not line.startswith('#')
		}


def compute_installed_deplocks(
	path_deps: Path,
) -> set[str]:
	return {
		conform_pypi_package_deplock(
			f'{dep.metadata["Name"]}=={dep.metadata["Version"]}'
		)
		for dep in importlib.metadata.distributions(path=[str(path_deps.resolve())])
	}


def deplock_conflicts(
	path_requirementslock: Path,
	path_deps: Path,
) -> list[str]:
	"""Check if packages defined in a 'requirements.lock' file are **strictly** realized by a particular dependency path.

	**Strict** means not only that everything is satisfied, but that _the exact versions_ are satisfied, and that _no extra packages_ are installed either.

	Parameters:
		path_requirementslock: Path to the `requirements.lock` file.
			Generally, one would use `ct.addon.PATH_REQS` to use the `requirements.lock` file shipped with the addon.
		path_deps: Path to the directory where Python modules can be found.
			Corresponds to the directory into which `pip install --target` was used to install packages.

	Returns:
		A list of messages explaining mismatches between the currently installed dependencies, and the given `requirements.lock` file.
		There are three kinds of conflicts:

		- **Version**: The wrong version of something is installed.
		- **Missing**: Something should be installed that isn't.
		- **Superfluous**: Something is installed that shouldn't be.
	"""
	required_deplocks = compute_required_deplocks(path_requirementslock)
	installed_deplocks = compute_installed_deplocks(path_deps)

	# Determine Diff of Required vs. Installed
	req_not_inst = required_deplocks - installed_deplocks
	inst_not_req = installed_deplocks - required_deplocks
	conflicts = {
		req.split('==')[0]: (req.split('==')[1], inst.split('==')[1])
		for req in req_not_inst
		for inst in inst_not_req
		if req.split('==')[0] == inst.split('==')[0]
	}

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
# - Passive PyDeps Checker
####################
def check_pydeps(path_requirementslock: Path, path_deps: Path):
	"""Check if all dependencies are satisfied without `deplock_conflicts()` conflicts, and update globals in response.

	Notes:
		Use of the globals `DEPS_OK` and `DEPS_ISSUES` should be preferred in general, since they are very fast to access.

		**Only**, use `check_pydeps()` after any operation something that might have changed the dependency status; both to check the result, but also to update the globals.

	Parameters:
		path_requirementslock: Path to the `requirements.lock` file.
			Generally, one would use `ct.addon.PATH_REQS` to use the `requirements.lock` file shipped with the addon.
		path_deps: Path to the directory where Python modules can be found.
			Corresponds to the directory into which `pip install --target` was used to install packages.

	Returns:
		A list of messages explaining mismatches between the currently installed dependencies, and the given `requirements.lock` file.
		There are three kinds of conflicts:

		- **Version**: The wrong version of something is installed.
		- **Missing**: Something should be installed that isn't.
		- **Superfluous**: Something is installed that shouldn't be.
	"""
	global DEPS_OK  # noqa: PLW0603
	global DEPS_ISSUES  # noqa: PLW0603
	global DEPS_REQ_DEPLOCKS  # noqa: PLW0603
	global DEPS_INST_DEPLOCKS  # noqa: PLW0603

	log.info(
		'Analyzing PyDeps at: %s',
		str(path_deps),
	)
	if len(issues := deplock_conflicts(path_requirementslock, path_deps)) > 0:
		log.info(
			'PyDeps Check Failed - adjust Addon Preferences for: %s', ct.addon.NAME
		)
		log.debug('%s', ', '.join(issues))
		log.debug('PyDeps Conflicts: %s', ', '.join(issues))

		DEPS_OK = False
		DEPS_ISSUES = issues
	else:
		log.info('PyDeps Check Succeeded - DEPS_OK and DEPS_ISSUES have been updated')
		DEPS_OK = True
		DEPS_ISSUES = []

	DEPS_REQ_DEPLOCKS = compute_required_deplocks(path_requirementslock)
	DEPS_INST_DEPLOCKS = compute_installed_deplocks(path_deps)
	return DEPS_OK
