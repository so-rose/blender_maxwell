from pathlib import Path

from . import info
from .nodeps.utils import simple_logger

simple_logger.sync_bootstrap_logging(
	console_level=info.BOOTSTRAP_LOG_LEVEL,
)

from . import nodeps, preferences, registration  # noqa: E402
from .nodeps.utils import pydeps  # noqa: E402

log = simple_logger.get(__name__)

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


####################
# - Load and Register Addon
####################
log.info('Loading Before-Deps BL_REGISTER')
BL_REGISTER__BEFORE_DEPS = [
	*nodeps.operators.BL_REGISTER,
	*preferences.BL_REGISTER,
]


def BL_REGISTER__AFTER_DEPS(path_deps: Path):
	log.info('Loading After-Deps BL_REGISTER')
	with pydeps.importable_addon_deps(path_deps):
		from . import node_trees, operators
	return [
		*operators.BL_REGISTER,
		*node_trees.BL_REGISTER,
	]


log.info('Loading Before-Deps BL_KEYMAP_ITEM_DEFS')
BL_KEYMAP_ITEM_DEFS__BEFORE_DEPS = [
	*nodeps.operators.BL_KEYMAP_ITEM_DEFS,
]


def BL_KEYMAP_ITEM_DEFS__AFTER_DEPS(path_deps: Path):
	log.info('Loading After-Deps BL_KEYMAP_ITEM_DEFS')
	with pydeps.importable_addon_deps(path_deps):
		from . import operators
	return [
		*operators.BL_KEYMAP_ITEM_DEFS,
	]


####################
# - Registration
####################
def register():
	"""Register the Blender addon."""
	log.info('Starting %s Registration', info.ADDON_NAME)

	# Register Barebones Addon for Dependency Installation
	registration.register_classes(BL_REGISTER__BEFORE_DEPS)
	registration.register_keymap_items(BL_KEYMAP_ITEM_DEFS__BEFORE_DEPS)

	# Retrieve PyDeps Path from Addon Preferences
	if (addon_prefs := info.addon_prefs()) is None:
		unregister()
		msg = f'Addon preferences not found; aborting registration of {info.ADDON_NAME}'
		raise RuntimeError(msg)
	log.debug('Found Addon Preferences')

	# Retrieve PyDeps Path
	path_pydeps = addon_prefs.pydeps_path
	log.info('Loaded PyDeps Path from Addon Prefs: %s', path_pydeps)

	if pydeps.check_pydeps(path_pydeps):
		log.info('PyDeps Satisfied: Loading Addon %s', info.ADDON_NAME)
		registration.register_classes(BL_REGISTER__AFTER_DEPS(path_pydeps))
		registration.register_keymap_items(BL_KEYMAP_ITEM_DEFS__AFTER_DEPS(path_pydeps))
	else:
		log.info(
			'PyDeps Invalid: Delaying Addon Registration of %s',
			info.ADDON_NAME,
		)
		registration.delay_registration(
			registration.EVENT__DEPS_SATISFIED,
			classes_cb=BL_REGISTER__AFTER_DEPS,
			keymap_item_defs_cb=BL_KEYMAP_ITEM_DEFS__AFTER_DEPS,
		)
		## TODO: bpy Popup to Deal w/Dependency Errors


def unregister():
	"""Unregister the Blender addon."""
	log.info('Starting %s Unregister', info.ADDON_NAME)
	registration.unregister_classes()
	registration.unregister_keymap_items()
	log.info('Finished %s Unregister', info.ADDON_NAME)
