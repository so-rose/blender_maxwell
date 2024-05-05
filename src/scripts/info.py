# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import tomllib
from pathlib import Path

PATH_ROOT = Path(__file__).resolve().parent.parent.parent
PATH_SRC = PATH_ROOT / 'src'

# Scripts
PATH_BL_DELETE_ADDON = PATH_SRC / 'scripts' / 'bl_delete_addon.py'
PATH_BL_INSTALL_ADDON = PATH_SRC / 'scripts' / 'bl_install_addon.py'
PATH_BL_RUN_DEV = PATH_SRC / 'scripts' / 'bl_run_dev.py'

# Build Dir
PATH_BUILD = PATH_ROOT / 'build'
PATH_BUILD.mkdir(exist_ok=True)

# Dev Dir
PATH_DEV = PATH_ROOT / 'dev'
PATH_DEV.mkdir(exist_ok=True)

####################
# - BL_RUN stdout Signals
####################
SIGNAL_START_CLEAN_BLENDER = 'SIGNAL__blender_is_clean'

####################
# - BL_RUN Exit Codes
####################
STATUS_NOCHANGE_ADDON = 42
STATUS_UNINSTALLED_ADDON = 42
STATUS_INSTALLED_ADDON = 69
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

# Set Bootstrap Log Level
## This will be the log-level of both console and file logs, at first...
## ...until the addon preferences have been loaded.
BOOTSTRAP_LOG_LEVEL = logging.DEBUG
BOOTSTRAP_LOG_LEVEL_FILENAME = '.bootstrap_log_level'

# Install the ZIPped Addon
####################
# - Development Information
####################
PATH_ADDON_DEV_BLEND = PATH_DEV / 'demo.blend'

PATH_ADDON_DEV_DEPS = PATH_DEV / '.cached-dev-dependencies'
PATH_ADDON_DEV_CACHE = PATH_DEV / '.dev-addon-cache'
PATH_ADDON_DEV_DEPS.mkdir(exist_ok=True)
