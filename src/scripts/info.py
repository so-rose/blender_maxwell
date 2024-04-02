import tomllib
from pathlib import Path

PATH_ROOT = Path(__file__).resolve().parent.parent.parent
PATH_SRC = PATH_ROOT / 'src'
PATH_BL_RUN = PATH_SRC / 'scripts' / 'bl_run.py'

PATH_BUILD = PATH_ROOT / 'build'
PATH_BUILD.mkdir(exist_ok=True)

PATH_DEV = PATH_ROOT / 'dev'
PATH_DEV.mkdir(exist_ok=True)

####################
# - BL_RUN stdout Signals
####################
SIGNAL_START_CLEAN_BLENDER = 'SIGNAL__blender_is_clean'

####################
# - BL_RUN Exit Codes
####################
STATUS_UNINSTALLED_ADDON = 42
STATUS_NOINSTALL_ADDON = 68

####################
# - Addon Information
####################
with (PATH_ROOT / 'pyproject.toml').open('rb') as f:
	PROJ_SPEC = tomllib.load(f)

ADDON_NAME = PROJ_SPEC['project']['name']
ADDON_VERSION = PROJ_SPEC['project']['version']

####################
# - Packaging Information
####################
PATH_ADDON_PKG = PATH_ROOT / 'src' / ADDON_NAME
PATH_ADDON_ZIP = PATH_ROOT / 'build' / (ADDON_NAME + '__' + ADDON_VERSION + '.zip')

PATH_ADDON_BLEND_STARTER = PATH_ADDON_PKG / 'blenders' / 'starter.blend'

BOOTSTRAP_LOG_LEVEL_FILENAME = '.bootstrap_log_level'

# Install the ZIPped Addon
####################
# - Development Information
####################
PATH_ADDON_DEV_BLEND = PATH_DEV / 'demo.blend'

PATH_ADDON_DEV_DEPS = PATH_DEV / '.cached-dev-dependencies'
PATH_ADDON_DEV_DEPS.mkdir(exist_ok=True)
