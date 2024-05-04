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
