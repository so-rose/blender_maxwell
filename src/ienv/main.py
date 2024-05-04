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

from pathlib import Path

import ienv
import rich.traceback

rich.traceback.install(show_locals=True)

if __name__ == '__main__':
	# Modify Import Machinery
	ienv_base_path: Path = Path(__file__).resolve().parent / 'addons'
	ienv.init(ienv_base_path, always_passthrough={'rich'})

	# Addon-Specific Imports Now Work
	print('Importing Addon 1')
	from addons import addon_1

	print('Importing Addon 2')
	from addons import addon_2

	# Test Addons
	print()
	print('Addon 1 Function: np.__file__:', addon_1.np_file_addon_1())
	print('Addon 2 Function: np.__file__:', addon_2.np_file_addon_2())
