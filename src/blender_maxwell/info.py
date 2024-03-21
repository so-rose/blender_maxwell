import tomllib
from pathlib import Path

import bpy

####################
# - Addon Info
####################
PATH_ADDON_ROOT = Path(__file__).resolve().parent

# Addon Information
## bl_info is filled with PROJ_SPEC when packing the .zip.
with (PATH_ADDON_ROOT / 'pyproject.toml').open('rb') as f:
	PROJ_SPEC = tomllib.load(f)

ADDON_NAME = PROJ_SPEC['project']['name']
ADDON_VERSION = PROJ_SPEC['project']['version']

# PyDeps Path Info
## requirements.lock is written when packing the .zip.
## By default, the addon pydeps are kept in the addon dir.
PATH_REQS = PATH_ADDON_ROOT / 'requirements.lock'
DEFAULT_PATH_DEPS = PATH_ADDON_ROOT / '.addon_dependencies'

# Logging Info
## By default, the addon file log writes to the addon dir.
## The initial .log_level contents are written when packing the .zip.
## Subsequent changes are managed by nodeps.utils.simple_logger.py.
DEFAULT_LOG_PATH = PATH_ADDON_ROOT / 'addon.log'
DEFAULT_LOG_PATH.touch(exist_ok=True)

PATH_BOOTSTRAP_LOG_LEVEL = PATH_ADDON_ROOT / '.bootstrap_log_level'
with PATH_BOOTSTRAP_LOG_LEVEL.open('r') as f:
	BOOTSTRAP_LOG_LEVEL = int(f.read().strip())

####################
# - Addon Getters
####################
def addon_prefs() -> bpy.types.AddonPreferences | None:
	if (addon := bpy.context.preferences.addons.get(ADDON_NAME)) is None:
		return None

	return addon.preferences
