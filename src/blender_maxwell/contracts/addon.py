import random
import tomllib
from pathlib import Path

import bpy
import bpy_restrict_state

PATH_ADDON_ROOT = Path(__file__).resolve().parent.parent
with (PATH_ADDON_ROOT / 'pyproject.toml').open('rb') as f:
	PROJ_SPEC = tomllib.load(f)
	## bl_info is filled with PROJ_SPEC when packing the .zip.

NAME = PROJ_SPEC['project']['name']
VERSION = PROJ_SPEC['project']['version']

####################
# - Assets
####################
PATH_ASSETS = PATH_ADDON_ROOT / 'assets'

####################
# - PyDeps Info
####################
PATH_REQS = PATH_ADDON_ROOT / 'requirements.lock'
DEFAULT_PATH_DEPS = PATH_ADDON_ROOT / '.addon_dependencies'
## requirements.lock is written when packing the .zip.
## By default, the addon pydeps are kept in the addon dir.

####################
# - Local Addon Cache
####################
ADDON_CACHE = PATH_ADDON_ROOT / '.addon_cache'
ADDON_CACHE.mkdir(exist_ok=True)


####################
# - Dynamic Addon Information
####################
def is_loading() -> bool:
	"""Checks whether the addon is currently loading.

	While an addon is loading, `bpy.context` is temporarily very limited.
	For example, operators can't run while the addon is loading.

	By checking whether `bpy.context` is limited like this, we can determine whether the addon is currently loading.

	Notes:
		Since `bpy_restrict_state._RestrictContext` is a very internal thing, this function may be prone to breakage on Blender updates.

		**Keep an eye out**!

	Returns:
		Whether the addon has been fully loaded, such that `bpy.context` is fully accessible.
	"""
	return isinstance(bpy.context, bpy_restrict_state._RestrictContext)


def operator(name: str, *operator_args, **operator_kwargs) -> None:
	# Parse Operator Name
	operator_namespace, operator_name = name.split('.')
	if operator_namespace != NAME:
		msg = f'Tried to call operator {operator_name}, but addon operators may only use the addon operator namespace "{operator_namespace}.<name>"'
		raise RuntimeError(msg)

	# Addon Not Loading: Run Operator
	if not is_loading():
		operator = getattr(getattr(bpy.ops, NAME), operator_name)
		operator(*operator_args, **operator_kwargs)
	else:
		msg = f'Tried to call operator "{operator_name}" while addon is loading'
		raise RuntimeError(msg)


def prefs() -> bpy.types.AddonPreferences | None:
	if (addon := bpy.context.preferences.addons.get(NAME)) is None:
		msg = 'Addon is not installed'
		raise RuntimeError(msg)

	return addon.preferences


####################
# - Logging Info
####################
DEFAULT_LOG_PATH = PATH_ADDON_ROOT / 'addon.log'
DEFAULT_LOG_PATH.touch(exist_ok=True)
## By default, the addon file log writes to the addon dir.
## The initial .log_level contents are written when packing the .zip.
## Subsequent changes are managed by nodeps.utils.simple_logger.py.

PATH_BOOTSTRAP_LOG_LEVEL = PATH_ADDON_ROOT / '.bootstrap_log_level'
with PATH_BOOTSTRAP_LOG_LEVEL.open('r') as f:
	BOOTSTRAP_LOG_LEVEL = int(f.read().strip())
