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

# noqa: INP001
import os
import subprocess
import sys
from pathlib import Path

import info


####################
# - Blender Runner
####################
def run_blender(
	py_script: Path | None,
	load_devfile: bool = False,
	headless: bool = True,
	monitor: bool = False,
):
	process = subprocess.Popen(
		[
			'blender',
			*(['--background'] if headless else []),
			*(
				[
					'--python',
					str(py_script),
				]
				if py_script is not None
				else []
			),
			*([info.PATH_ADDON_DEV_BLEND] if load_devfile else []),
		],
		env=os.environ | {'PYTHONUNBUFFERED': '1'},
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)
	output = []
	printing_live = monitor

	# Process Real-Time Output
	for line in iter(process.stdout.readline, b''):
		if not line:
			break

		if printing_live:
			print(line, end='')  # noqa: T201
		elif (
			info.SIGNAL_START_CLEAN_BLENDER in line
			# or 'Traceback (most recent call last)' in line
		):
			printing_live = True
			print(''.join(output))  # noqa: T201
		else:
			output.append(line)

	# Wait for the process to finish and get the exit code
	process.wait()
	return process.returncode, output


####################
# - Main
####################
if __name__ == '__main__':
	# Uninstall Addon
	print(f'Blender: Uninstalling "{info.ADDON_NAME}"...')
	return_code, output = run_blender(info.PATH_BL_DELETE_ADDON, monitor=False)
	if return_code == info.STATUS_UNINSTALLED_ADDON:
		print(f'\tBlender: Uninstalled "{info.ADDON_NAME}"')
	elif return_code == info.STATUS_NOCHANGE_ADDON:
		print(f'\tBlender: "{info.ADDON_NAME}" Not Installed')

	# Install Addon
	print(f'Blender: Installing & Enabling "{info.ADDON_NAME}"...')
	return_code, output = run_blender(info.PATH_BL_INSTALL_ADDON, monitor=False)
	if return_code == info.STATUS_INSTALLED_ADDON:
		print(f'\tBlender: Install & Enable "{info.ADDON_NAME}"')
	else:
		print(f'\tBlender: "{info.ADDON_NAME}" Not Installed')
		print(*output, sep='')
		sys.exit(1)

	# Run Addon
	print(f'Blender: Running "{info.ADDON_NAME}"...')
	return_code, output = run_blender(
		None, headless=False, load_devfile=True, monitor=True
	)
