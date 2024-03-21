# noqa: INP001
import os
import subprocess
from pathlib import Path

import info


####################
# - Blender Runner
####################
def run_blender(py_script: Path, print_live: bool = False):
	process = subprocess.Popen(
		['blender', '--python', str(py_script)],
		env=os.environ | {'PYTHONUNBUFFERED': '1'},
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)
	output = []
	printing_live = print_live

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
# - Run Blender w/Clean Addon Reinstall
####################
def main():
	return_code, output = run_blender(info.PATH_BL_RUN, print_live=False)
	if return_code == info.STATUS_UNINSTALLED_ADDON:
		return_code, output = run_blender(info.PATH_BL_RUN, print_live=True)
		if return_code == info.STATUS_NOINSTALL_ADDON:
			msg = f"Couldn't install addon {info.ADDON_NAME}"
			raise ValueError(msg)
	elif return_code != 0:
		print(''.join(output))  # noqa: T201

if __name__ == "__main__":
	main()
