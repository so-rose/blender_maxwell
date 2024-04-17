import tomllib
from pathlib import Path

import bpy

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
# - Addon Prefs Info
####################
def prefs() -> bpy.types.AddonPreferences | None:
	if (addon := bpy.context.preferences.addons.get(NAME)) is None:
		return None

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
