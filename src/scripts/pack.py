# noqa: INP001

import contextlib
import logging
import tempfile
import typing as typ
import zipfile
from pathlib import Path

import info

LogLevel: typ.TypeAlias = int

_PROJ_VERSION_STR = str(
	tuple(int(el) for el in info.PROJ_SPEC['project']['version'].split('.'))
)
_PROJ_DESC_STR = info.PROJ_SPEC['project']['description']

BL_INFO_REPLACEMENTS = {
	"'version': (0, 0, 0),": f"'version': {_PROJ_VERSION_STR},",
	"'description': 'Placeholder',": f"'description': '{_PROJ_DESC_STR}',",
}


@contextlib.contextmanager
def zipped_addon(  # noqa: PLR0913
	path_addon_pkg: Path,
	path_addon_zip: Path,
	path_pyproject_toml: Path,
	path_requirements_lock: Path,
	initial_log_level: LogLevel = logging.INFO,
	replace_if_exists: bool = False,
	remove_after_close: bool = True,
) -> typ.Iterator[Path]:
	"""Context manager exposing a folder as a (temporary) zip file.

	Parameters:
		path_addon_pkg: Path to the folder containing __init__.py of the Blender addon.
		path_addon_zip: Path to the Addon ZIP to generate.
		path_pyproject_toml: Path to the `pyproject.toml` of the project.
			This is made available to the addon, to de-duplicate definition of name,
	The .zip file is deleted afterwards, unless `remove_after_close` is specified.
	"""
	# Delete Existing ZIP (maybe)
	if path_addon_zip.is_file():
		if replace_if_exists:
			msg = 'File already exists where ZIP would be made'
			raise ValueError(msg)
		path_addon_zip.unlink()

	# Create New ZIP file of the addon directory
	with zipfile.ZipFile(path_addon_zip, 'w', zipfile.ZIP_DEFLATED) as f_zip:
		# Install Addon Files @ /*
		for file_to_zip in path_addon_pkg.rglob('*'):
			# Dynamically Alter 'bl_info' in __init__.py
			## This is the only way to propagate ex. version information
			if str(file_to_zip.relative_to(path_addon_pkg)) == '__init__.py':
				with (
					file_to_zip.open('r') as f_init,
					tempfile.NamedTemporaryFile(mode='w') as f_tmp,
				):
					initpy = f_init.read()
					for (
						to_replace,
						replacement,
					) in BL_INFO_REPLACEMENTS.items():
						initpy = initpy.replace(to_replace, replacement)
					f_tmp.write(initpy)

					# Write to ZIP
					f_zip.writestr(
						str(file_to_zip.relative_to(path_addon_pkg.parent)),
						initpy,
					)

			# Write File to Zip
			else:
				f_zip.write(file_to_zip, file_to_zip.relative_to(path_addon_pkg.parent))

		# Install pyproject.toml @ /pyproject.toml of Addon
		f_zip.write(
			path_pyproject_toml,
			str(Path(path_addon_pkg.name) / Path(path_pyproject_toml.name)),
		)

		# Install requirements.lock @ /requirements.txt of Addon
		f_zip.write(
			path_requirements_lock,
			str(Path(path_addon_pkg.name) / Path(path_requirements_lock.name)),
		)

		# Set Initial Log-Level
		f_zip.writestr(
			str(Path(path_addon_pkg.name) / info.BOOTSTRAP_LOG_LEVEL_FILENAME),
			str(initial_log_level),
		)

	# Delete the ZIP
	try:
		yield path_addon_zip
	finally:
		if remove_after_close:
			path_addon_zip.unlink()


####################
# - Run Blender w/Clean Addon Reinstall
####################
def main():
	with zipped_addon(
		path_addon_pkg=info.PATH_ADDON_PKG,
		path_addon_zip=info.PATH_ADDON_ZIP,
		path_pyproject_toml=info.PATH_ROOT / 'pyproject.toml',
		path_requirements_lock=info.PATH_ROOT / 'requirements.lock',
		replace_if_exists=True,
		remove_after_close=False,
	):
		# TODO: GPG signature for distribution
		pass


if __name__ == '__main__':
	main()
