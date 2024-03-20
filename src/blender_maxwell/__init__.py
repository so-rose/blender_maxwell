import tomllib
from pathlib import Path

import bpy

from . import operators_nodeps, preferences, registration
from .utils import pydeps
from .utils import logger as _logger

log = _logger.get()
PATH_ADDON_ROOT = Path(__file__).resolve().parent
with (PATH_ADDON_ROOT / 'pyproject.toml').open('rb') as f:
	PROJ_SPEC = tomllib.load(f)

####################
# - Addon Information
####################
# The following parameters are replaced when packing the addon ZIP
## - description
## - version
bl_info = {
	'name': 'Maxwell PDE Sim and Viz',
	'blender': (4, 1, 0),
	'category': 'Node',
	'description': 'Placeholder',
	'author': 'Sofus Albert Høgsbro Rose',
	'version': (0, 0, 0),
	'wiki_url': 'https://git.sofus.io/dtu-courses/bsc_thesis',
	'tracker_url': 'https://git.sofus.io/dtu-courses/bsc_thesis/issues',
}
## bl_info MUST readable via. ast.parse
## See scripts/pack.py::BL_INFO_REPLACEMENTS for active replacements
## The mechanism is a 'dumb' - output of 'ruff fmt' MUST be basis for replacing


def ADDON_PREFS():
	return bpy.context.preferences.addons[
		PROJ_SPEC['project']['name']
	].preferences


####################
# - Load and Register Addon
####################
BL_REGISTER__BEFORE_DEPS = [
	*operators_nodeps.BL_REGISTER,
	*preferences.BL_REGISTER,
]


def BL_REGISTER__AFTER_DEPS(path_deps: Path):
	with pydeps.importable_addon_deps(path_deps):
		from . import node_trees, operators
	return [
		*operators.BL_REGISTER,
		*node_trees.BL_REGISTER,
	]


def BL_KEYMAP_ITEM_DEFS(path_deps: Path):
	with pydeps.importable_addon_deps(path_deps):
		from . import operators
	return [
		*operators.BL_KMI_REGISTER,
	]


####################
# - Registration
####################
def register():
	# Register Barebones Addon for Dependency Installation
	registration.register_classes(BL_REGISTER__BEFORE_DEPS)

	# Retrieve PyDeps Path from Addon Preferences
	addon_prefs = ADDON_PREFS()
	path_pydeps = addon_prefs.path_addon_pydeps

	# If Dependencies are Satisfied, Register Everything
	if pydeps.check_pydeps(path_pydeps):
		registration.register_classes(BL_REGISTER__AFTER_DEPS())
		registration.register_keymap_items(BL_KEYMAP_ITEM_DEFS())
	else:
		# Delay Registration
		registration.delay_registration(
			registration.EVENT__DEPS_SATISFIED,
			classes_cb=BL_REGISTER__AFTER_DEPS,
			keymap_item_defs_cb=BL_KEYMAP_ITEM_DEFS,
		)

		# TODO: A popup before the addon fully loads or something like that?
		## TODO: Communicate that deps must be installed and all that?


def unregister():
	registration.unregister_classes()
	registration.unregister_keymap_items()
