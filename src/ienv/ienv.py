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

"""Interpreter-integrated ENVironments - like 'venv', but in one same Python process!

Ever wanted to **robustly** use two subpackages, with their own dependencies, in the same process?
Now you can, by letting the package be an 'ienv'!

The cost is your soul, of course.
Well, a lightly customized `builtins.__import__`, but isn't that the same kind of deal?


# Example
Let's presume you've setup your project structure something like this:

```
main.py  <-- Run this (`ienv` must be importable).

children/
.. child1/  <-- This is an IEnv
.. child1/.ienv-deps
.. child1/__init__.py

.. child2/  <-- This is also an IEnv
.. child2/.ienv-deps
.. child2/__init__.py
```

Say you want to run the following `main.py`, which prints out the `__file__` attribute of `numpy` imported in each:
```
from children import child1
from children import child2

print('Addon 1 Function: np.__file__:', addon_1.np_file_addon_1())
print('Addon 2 Function: np.__file__:', addon_2.np_file_addon_2())
```

However, your boss says that:
- `child1` **must** use `numpy==1.24.4`
- `child2` **must** use `numpy==1.26.1`

Generally, this would be impossible.
But that's where IEnv comes in.

### Installing `ienv.py`
It's the usual story: As long as `main.py` can `import ienv`, you're set.

Some ideas:
- A `venv`: This is the recommended setup.
- The same folder as `main.py`: If you run `python ./main.py`, then you're set.
- Any `sys.path` Folder: The general case.

`ienv.py` has no dependencies, so it should be perfectly portable to all kinds of weird setups.

### Installing Dependencies to the IEnvs
Let's quickly install numpy on each.
- `python -m pip install --target child1/.ienv numpy==1.24.4`.
- `python -m pip install --target child2/.ienv numpy==1.26.1`.

**Make sure to use the same `python` as you'll be running `main.py` with.**

_You could also do this from within `main.py`, with the help of `subprocess.run`._

### Run `main.py`
To run the main.py, we just need to add a little snippet above everything else:

```
import ienv
from pathlib import Path

ienv_base_path = Path(__file__).resolve().parent / 'children'
ienv.init(ienv_base_path)

...
```

Now, when you run `main.py`, you should see a very pro



# IEnv Semantics
**An "IEnv" is a Python package with its own dependencies.**
What's special is that **IEnvs can share a process without sharing dependencies**.

This all happens without the code in the IEnv having to do anything special.

## Classification
To be classified as an IEnv, a Python module:
- **MUST** be a valid Python package, with an `__init__.py`.
- **CANNOT** be the entrypoint of the program.
- **MUST** be imported from a context where `ienv.init()` has been run.
- **MUST** be a subfolder of the `ienv_base_path` passed as an argument to the latest run of `ienv.init()`.
- **MUST** have a subfolder named `.ienv-deps`, which only contains Python modules (incl. packages).

## General Behavior
From any module in IEnv (or the IEnv itself), `import` will now work slightly differently:
- `import` will prioritize searching `.ienv-deps` (and can be configured to reject other sources).
- If a module is found in `.ienv-deps`, the `sys.modules` module name will have an IEnv-specific prefix.
- `import` will always check `sys.modules` using the IEnv-prefixed name.

It's just as important what `ienv` **DOES NOT** do:
- All `stdlib` imports are passed through to the builtin `__import__`.
- The user may also specify modules to always pass through.
- The performance properties of `sys.modules` are completely preserved, even within IEnvs.


# Gotchas
There are some **gotchas** you must make peace with if you want to use IEnvs.

## No Dynamic imports
**Dynamic imports are not guaranteed available after `ienv.free()` has run.**
Don't use them!

Note that:
- They're generally a bad idea anyway, as **the import semantics of dynamic contexts cannot be statically known**.
- If your program never runs `ienv.free()`, then dynamic imports will work just fine.

## Not Portable
**`pip install`ed packages can never be presumed portable across operating systems**.
As a result, IEnvs are not generally copy-pasteable to other folders or operating systems.

Note that:
- If you're certain that no dependencies will break by being moved from their install dir, then the IEnv can be moved.
- If, also, all dependencies are cross-platform, then the IEnv can be copied to other platforms.

Python modules, being very dynamic, may have undefined behavior in response to being moved.
"""

import builtins
import dataclasses
import enum
import functools
import importlib
import importlib.abc
import importlib.machinery
import importlib.util  ## This should already make you concerned :)
import os
import re
import sys
import types
import typing as typ
from pathlib import Path

builtins__import = __import__

####################
# - Types
####################
ValidDirName: typ.TypeAlias = str
PathLikeStr: typ.TypeAlias = str
ModuleNamePrefix: typ.TypeAlias = str
ModuleName: typ.TypeAlias = str
AbsoluteModuleName: typ.TypeAlias = str
IEnvName: typ.TypeAlias = str

####################
# - IEnv Constants
####################
_USE_CPYTHON_MODULE_SUFFIX_PRECEDENCE: bool = False

_IENV_PREFIX: ModuleNamePrefix = '_ienv_'
_IENV_DEPS_DIRNAME: ValidDirName = '.ienv-deps'
IENV_BASE_PATH: Path | None = None
ALWAYS_PASSTHROUGH: set[ModuleName] | None = None


####################
# - IEnv Analysis Functions
####################
@functools.cache
def is_in_ienv(caller_path_str: PathLikeStr) -> bool:
	return IENV_BASE_PATH in Path(caller_path_str).parents


@functools.cache
def compute_ienv_name(caller_path: Path) -> IEnvName:
	if not is_in_ienv(os.fspath(caller_path)):  ## Reuse @cache by stringifying Path
		msg = f'Attempted to import an IEnv, but caller ({caller_path}) is not in the IENV_BASE_PATH ({IENV_BASE_PATH})'
		raise ImportError(msg)

	return caller_path.relative_to(IENV_BASE_PATH).parts[0]


@functools.cache
def compute_ienv_path(ienv_name: IEnvName) -> Path:
	return IENV_BASE_PATH / ienv_name


@functools.cache
def compute_ienv_deps_path(ienv_name: IEnvName) -> Path:
	return IENV_BASE_PATH / ienv_name / _IENV_DEPS_DIRNAME


@functools.cache
def compute_ienv_module_prefix(ienv_name: IEnvName) -> ModuleNamePrefix:
	return _IENV_PREFIX + f'{ienv_name}__'


@functools.cache
def match_ienv_module_name(ienv_module_name: AbsoluteModuleName) -> re.Match | None:
	return re.match(r'^_ienv_(?P<ienv_name>[a-z0-9\_-]+)__', ienv_module_name)


####################
# - IEnv __import__
####################
def import_ienv(
	name: str,
	_globals: dict[str, typ.Any] | None = None,
	_locals: dict[str, typ.Any] | None = None,
	fromlist: tuple[str, ...] = (),
	level: int = 0,
) -> types.ModuleType:
	"""Imports an `ienv`, using the same context provided to `__import__`.

	# Semantics
	This function is designed to be called from a replaced `builtins.__import__`.
	Thus, its semantics are identical to `__import__`, but differs in exactly two subtle ways.

	**Namespaced `sys.modules` Lookup**
	- Usually, `import name` will lookup 'name' in `sys.modules`.
	- Now, `import name` will lookup '_ienv_<ienv_name>__<name>' in `sys.modules`.

	**Namespaced `sys.modules` Assignment**
	- Usually, `import name` -> `sys.modules['name']`.
	- Now, `import name` -> `sys.modules['_ienv_<ienv_name>__<name>']`

	## Relationship to `sys.meta_path` Finder
	Strictly speaking, the second one (**Assignment**) is performed by a complementary `sys.meta_path` finder.
	However, this finder only triggers when `builtins.__import__` is called with a specially-prefixed name.
	This function automates the preparation of this specially-prefixed name.

	Arguments:
		name: The name of the module to import.
		_globals: The `globals()` dictionary from where `import` was called.
			This is used to decide which module to import and return.
		_locals: The `globals()` dictionary from where `import` was called.
			As with `builtins.__import__`, it must be defined, but it is not used.
			It is included here (and passed on) to match these semantics.
		fromlist: Names to guarantee available in the returned module.
			For each `attr in fromlist`, it must be possible to call `mod.attr` on the returned module `mod`.
		level: The amount of module nesting.
			Always `>= 0`.
			`level=0` denotes an absolute import, ex. `import name`.
			`level>0` denotes a relative import, ex. `from ... import name`.
			For more details, see the source code.

	Returns:
		An imported module, referring to the same object an an IEnv-namespaced `sys.modules` entry.

	Raises:
		ImportError: Cannot be called from any module not within an IEnv path.
	"""
	# Scan Caller for Context
	## _globals contains all information for how to import.
	caller_package: str | None = _globals.get('__package__')

	# Compute IEnv Name
	## From Caller __file__
	if '__file__' in _globals:
		ienv_name = compute_ienv_name(Path(_globals['__file__']))

	## From Caller __name__
	### This makes dynamic imports from IEnv modules also IEnv-namespaced.
	elif (
		'__name__' in _globals
		and _globals['__name__'].startswith(_IENV_PREFIX)
		and (_match := match_ienv_module_name(_globals['__name__'].split('.')[0]))
	):
		ienv_name = _match['ienv_name']

	## Caller Invalid
	else:
		msg = 'An IEnv import was attempted where neither __file__ nor __name__ are present in the caller globals()'
		raise RuntimeError(msg)

	# Compute IEnv Module Prefix
	ienv_module_prefix = compute_ienv_module_prefix(ienv_name)

	# Compute Absolute Module Name
	## '.' is folder separator.
	## Top-level module is in a sys.path-searchable folder.
	importing_submodule = False
	if level == 0:
		# Absolute Name is Top-Level Module
		## -> 'import module.var1' (only imports module)
		if '.' in name and len(fromlist) == 0:
			abs_import_name = name.split('.')[0]

		# INVALID: Top-Level Relative Import
		## -> 'import .' (invalid syntax)
		elif name == '':
			msg = f'Caller attempted a top-level relative import (caller package={caller_package})'
			raise ImportError(msg)

		# Absolute Name is Name (any of the following)
		## len(fromlist) == 0 -> 'import module'
		## len(fromlist) > 0 -> 'from module import var1, ...'
		## len(fromlist) > 0 -> 'from module1.module2 import var1, ...'
		else:
			abs_import_name = name

	elif level > 0:
		if caller_package is None:
			msg = 'Caller attempted a relative import, but has no __package__'
			raise ImportError(msg)

		# Absolute Name is Current Package
		## -> 'from . import var1, ...'
		if name == '' and len(fromlist) > 0:
			abs_import_name = caller_package

		# INVALID:
		## -> 'from .' (invalid syntax)
		elif name == '' and len(fromlist) == 0:
			msg = 'Caller attempted to import nothing from current package ({caller_package})'
			raise ImportError(msg)

		# Absolute Name is Package and Module
		## -> 'from ...spam.ham import var1, ...'
		elif name == '' and len(fromlist) > 0:
			abs_import_name = '.'.join([caller_package, name])

		# Absolute Name is Module
		## -> 'from spam import var1, ...'
		elif len(fromlist) > 0:
			abs_import_name = name
			importing_submodule = True

		# INVALID: Top-Level Module is Relative
		## -> 'import .module.var1'
		elif '.' in name and len(fromlist) == 0:
			msg = f'Caller attempted to import its own package ({caller_package})'
			raise ImportError(msg)

	# Compute (Absolute) Module Name w/wo IEnv-Specific Prefix
	## Imported with Non-IEnv-Prefixed Name
	if importing_submodule:
		print(abs_import_name)
		# print(sys.modules)
	if not abs_import_name.startswith(ienv_module_prefix) and not importing_submodule:
		# module_name = abs_import_name
		ienv_module_name = ienv_module_prefix + abs_import_name

	## Imported with IEnv-Prefixed Name
	else:
		# module_name = abs_import_name.removeprefix(ienv_module_prefix)
		ienv_module_name = abs_import_name

	# Lookup IEnv-Prefixed (Absolute) Module Name in sys.modules
	## This preserves the caching behavior of __import__.
	## This snippet is the ONLY reason to override __import__.
	if (_module := sys.modules.get(ienv_module_name)) is not None:
		return _module

	# Import IEnv-Prefixed (Absolute) Module Name
	## The builtin __import__ statement will use 'sys.meta_path' to import the module.
	## We've injected a custom "Finder" into 'sys.meta_path'.
	## Our custom "Finder" will ensure that 'sys.modules' is filled with 'ienv_module_name'.
	return builtins__import(
		ienv_module_name,
		globals=_globals,
		locals=_locals,
		fromlist=fromlist,
		level=level,
	)


####################
# - __import__ Replacement
####################
def _import(
	name,
	globals=None,  # noqa: A002
	locals=None,  # noqa: A002
	fromlist=(),
	level=0,
) -> types.ModuleType:
	if (
		## Never Hijack stdlib Imports
		name not in sys.stdlib_module_names
		## Never Hijack "Special" Imports
		and name not in ALWAYS_PASSTHROUGH
		## Only Hijack if Caller has Globals
		and globals is not None
		## Hijack if Caller in IEnv (determined by __file__ or __name__)
		and (
			# Detect that Caller is in IEnv by __file__
			'__file__' in globals
			and is_in_ienv(globals['__file__'])
			# Detect that Caller is in IEnv by __package__ == __name__
			## __init__.py may not have __file__; this is how we detect that.
			# or (
			# '__file__' not in globals
			# and '__package__' in globals
			# and '__name__' in globals
			# and globals['__name__'] == globals['__package__']
			# and globals['__name__'].startswith(_IENV_PREFIX)
			# )
			or (
				'__file__' not in globals
				and '__path__' in globals
				and len(globals['__path__']) > 0
				and is_in_ienv(globals['__path__'][0])
			)
		)
	):
		return import_ienv(
			name, _globals=globals, _locals=locals, fromlist=fromlist, level=level
		)

	return builtins__import(
		name, globals=globals, locals=locals, fromlist=fromlist, level=level
	)


# _ArrayFunctionDispatcher
####################
# - IEnv Module Info
####################
class ModuleType(enum.StrEnum):
	Source = enum.auto()  ## File w/Python Code (.py)
	Bytecode = enum.auto()  ## File w/Python Bytecode (.pyc)
	Extension = enum.auto()  ## Compiled Extension Module (.so/.dll)
	Package = enum.auto()  ## Folder w/__init__.py
	Namespace = enum.auto()  ## Folder w/o _-init__.py
	Builtin = enum.auto()  ## stdlib Modules (compiled into the Python interpreter)
	Frozen = enum.auto()  ## Compiled into Python interpreter


# ModuleType to Loader Mapping
## Almost identical call signatures:
## - SourceFileLoader: (fullname, path)
## - SourcelessFileLoader: (fullname, path)
## - ExtensionFileLoader: (fullname, path)
## - BuiltinImporter: ()
## - Frozen: ()
_MODULE_LOADERS: dict[ModuleType, importlib.abc.Loader] = {
	ModuleType.Source: importlib.machinery.SourceFileLoader,
	ModuleType.Bytecode: importlib.machinery.SourcelessFileLoader,
	ModuleType.Extension: importlib.machinery.ExtensionFileLoader,
	ModuleType.Package: importlib.machinery.SourceFileLoader,  ## Load __init__.py
	ModuleType.Namespace: None,
	ModuleType.Builtin: importlib.machinery.BuiltinImporter,
	ModuleType.Frozen: importlib.machinery.FrozenImporter,
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class IEnvModuleInfo:
	"""Information about a module that can be depended on by an IEnv.

	Based on the IEnv-specific name of the module, information about the IEnv that depends on this module can be computed.
	Such information is available as computed properties.

	This module is always associated with a subpath of `ienv_deps_path`.
	In particular, the module always has a ModuleType of one of:
	- ModuleType.Source
	- ModuleType.Bytecode
	- ModuleType.Extension
	- ModuleType.Package
	- ModuleType.Namespace

	"""

	ienv_module_name: AbsoluteModuleName

	####################
	# - IEnv Info wrt. Module
	####################
	@functools.cached_property
	def ienv_name(self) -> IEnvName:
		if match := match_ienv_module_name(self.ienv_module_name):
			return match['ienv_name']

		msg = f'Parsing IEnv Name from Module "{self.ienv_module_name}" failed; is the module prefixed with "{_IENV_PREFIX}"?'
		raise RuntimeError(msg)

	@property
	def ienv_prefix(self) -> ModuleNamePrefix:
		return compute_ienv_module_prefix(self.ienv_name)

	@property
	def ienv_deps_path(self) -> Path:
		return compute_ienv_deps_path(self.ienv_name)

	####################
	# - Module Info
	####################
	@functools.cached_property
	def module_name(self) -> AbsoluteModuleName:
		return self.ienv_module_name.removeprefix(self.ienv_prefix)

	@functools.cached_property
	def module_path(self) -> Path:
		"""Computes the path to this module, guaranteeing that it is either a directory or a file.

		When the module is a file, all supported module suffixes are tested.
		If no files with a supported module suffix match an existing file, then an `ImportError` is thrown.

		If more than one file exists at the path with a supported module suffix, we're left with a question of "module suffix precedence".
		There are two philosophies about how to deal with this:
		- SystemError (default): Since the Python language doesn't specify which file to load, the choice is ambiguous, and the program cannot continue. We should therefore throw an explicit SystemError to encourage users to complain about the lack of specification. **May break some libraries** (but maybe they shouldn't work to begin with)
		- CPython Precedence: Since Python has a de-facto implementation, we should fallback to its behavior. In `importlib/_bootstrap_external.py` we can clearly see, in `_get_supported_file_loaders()`, that the precedence goes (highest to lowest): **Extensions, source, bytecode**.

		Use the module-level `_USE_CPYTHON_MODULE_SUFFIX_PRECEDENCE` global variable to select which behavior you prefer.
		**Note that "CPython Precedence" will NOT try to match CPython's precedence within each category of suffix.**

		Returns:
			The path to the module itself

		Raises:
			ImportError: The computed path isn't a directory, and NO file exists at the path with a supported module suffix.
			SystemError: The computed path isn't a directory, >1 file could potentially be imported, and CPython module suffix precedence is not in use.
		"""
		# Load the Module Path w/o Extension
		module_path_noext = self.ienv_deps_path / Path(*self.module_name.split('.'))

		# Is Directory: Directories Don't Have FILE Extensions!
		if module_path_noext.is_dir():
			return module_path_noext

		module_path_candidates = [
			module_path_candidate
			for module_suffix in importlib.machinery.all_suffixes()
			if (
				module_path_candidate := module_path_noext.with_suffix(module_suffix)
			).is_file()
		]
		if len(module_path_candidates) == 1:
			return module_path_candidates[0]
		if len(module_path_candidates) == 0:
			msg = f'Computed module base path {module_path_noext} for {self.ienv_module_name} does not have a file with a valid module extension'
			raise ImportError(msg)

		# >1 Module Path Candidates
		## We can choose to approximate CPython's module suffix precedence.
		## Or, we throw an error, since module choice is ambiguous.
		if _USE_CPYTHON_MODULE_SUFFIX_PRECEDENCE:
			module_path_candidates.sort(
				key=lambda el: (
					3 * int(el.suffix in importlib.machinery.EXTENSION_SUFFIXES)
					+ 2 * int(el.suffix in importlib.machinery.SOURCE_SUFFIXES)
					+ 1 * int(el.suffix in importlib.machinery.BYTECODE_SUFFIXES)
				)
			)
			return module_path_candidates[0]

		msg = f'Computed module base path {module_path_noext} for {self.ienv_module_name} does not have ONE, unambiguous file from which to load a module; it has {len(module_path_candidates)}'
		raise SystemError(msg)

	@functools.cached_property
	def module_type(self) -> ModuleType:
		"""Computes the type of this module.

		Raises:
			ValueError: If the suffix of the module path doesn't indicate a valid Python module.
			RuntimeError: If the module path couldn't matched to a module type, or the module path is no longer a directory or file.
				`self.module_path` should guarantee that the module path is either a directory or a file.
		"""
		# Module Path is Directory: Package or Namespace
		if self.module_path.is_dir():
			# Module is Package
			if (self.module_path / '__init__.py').is_file():
				return ModuleType.Package

			# Module is Namespace
			return ModuleType.Namespace

		if self.module_path.is_file():
			module_file_extension = ''.join(self.module_path.suffixes)
			if module_file_extension not in importlib.machinery.all_suffixes():
				msg = f"The file {self.module_path} has a suffix {module_file_extension} which the current Python process doesn't recognize as a valid Python module extension. Is the file extension compatible with the current OS?"
				raise ValueError(msg)

			# Module is Source File
			if module_file_extension in importlib.machinery.SOURCE_SUFFIXES:
				return ModuleType.Source

			# Module is Bytecode
			if module_file_extension in importlib.machinery.BYTECODE_SUFFIXES:
				return ModuleType.Bytecode

			# Module is Compiled Extension
			if module_file_extension in importlib.machinery.EXTENSION_SUFFIXES:
				return ModuleType.Extension

			msg = f'Module {self.module_path} refers to a valid module file in this context, but the suffix {module_file_extension} could not be matched to a known module type. Please contact the author of IEnv'
			raise RuntimeError(msg)

		msg = f"Computed module path {self.module_path} is neither a directory or a file. This shouldn't happen; most likely, the path was changed by another process"
		raise RuntimeError(msg)

	####################
	# - IEnv Module Spec/Loader
	####################
	@functools.cached_property
	def module_source_path(self) -> Path:
		if self.module_type == ModuleType.Package:
			return self.module_path / '__init__.py'
		if self.module_type == ModuleType.Namespace:
			return None

		return self.module_path

	@property
	def module_loader(self) -> Path:
		"""Selects an appropriate loader for this module."""
		return _MODULE_LOADERS[self.module_type](
			self.ienv_module_name, os.fspath(self.module_source_path)
		)

	@property
	def module_spec(self) -> importlib.machinery.ModuleSpec:
		"""Construct a ModuleSpec with appropriate attributes.

		We select module attributes via the ModuleSpec constructor, according to the following understanding of Python's import semantics.

		ModuleSpec -> __spec__
			Controls the entire import process of a module.
			Its attributes set the module attributes.
			When __spec__.parent is undefined, __package__ is used.
			__main__ has a special __spec__, which might be None.
		name -> __name__
			Identifies the module in sys.modules.
		loader -> __loader__
			Actually loads the module on import.
		origin -> __file__
			Path to the file from which this module is loaded.
			If the module isn't loaded from a file, this is None.
			MUST be 'None' for Namespace modules.
			NEVER defined for Builtin/Frozen modules.
			MAY be left undefined for domain-specific reasons.
		submodule_search_locations -> __path__
			ONLY set for package modules (and may be empty).
			(The DEFINITION of a package is "a module with __path__")
			In this context, namespace packages are "packages".
		loader_state
			Module-specific data provided to the loader.
			Unused.
		cached -> __cached__
			MAY be defined IF __file__ is defined.
			Path to a compiled version of __file__.
			Doesn't have to point to a path that exists.
			MAY be set without __file__, but this is atypical.
			Can be set to None if compiled code isn't used.
		parent -> __package__
			For __init__.py, this is the same as 'name'
			For top-level modules, this is ''.
			Else, this is the absolute path to the module's parent package.
			When __package__ is undefined, __spec__.parent is used.
		has_location
			When True, 'origin' is a loadable location.
			When False, it is not.
			Note, this is merely a hint given to the Loader.
		is_package
			Following InspectLoader.is-package, namespace packages are not "packages".
		"""
		spec = importlib.machinery.ModuleSpec(
			self.ienv_module_name,  ## __name__
			self.module_loader,  ## __loader__
			origin=os.fspath(self.module_source_path),
			loader_state=None,
			is_package=self.module_type == ModuleType.Package,
		)
		spec.submodule_search_locations = (
			[os.fspath(self.module_path)]
			if self.module_type in {ModuleType.Package, ModuleType.Namespace}
			else None
		)
		spec.cached = None
		print(spec)
		print('SEARCH', spec.submodule_search_locations)
		# print(spec.loader.name)
		# print(spec.loader.path)
		return spec


####################
# - sys.meta_path Finder
####################
class IEnvMetaPathFinder(importlib.abc.MetaPathFinder):
	@staticmethod
	def find_spec(
		fullname: str,
		path: str | None,  # noqa: ARG004
		target: types.ModuleType | None = None,  # noqa: ARG004
	) -> importlib.machinery.ModuleSpec | None:
		"""When the import 'fullname' has the IEnv prefix, load the module from the IEnv deps path."""
		if fullname.startswith(_IENV_PREFIX):
			mod_info = IEnvModuleInfo(ienv_module_name=fullname)
			return mod_info.module_spec

		# Pass to Next MetaPathFinder
		return None


####################
# - Initialization
####################
def init(
	ienv_base_path: Path,
	always_passthrough: set[ModuleName] = frozenset(),
):
	"""Initialize IEnv handling."""
	global IENV_BASE_PATH, ALWAYS_PASSTHROUGH  # noqa: PLW0603
	IENV_BASE_PATH = ienv_base_path
	ALWAYS_PASSTHROUGH = always_passthrough

	is_in_ienv.cache_clear()
	compute_ienv_name.cache_clear()
	compute_ienv_deps_path.cache_clear()
	## compute_ienv_module_prefix uses no globals

	# Modify Builtins
	## You can always get the original back via 'ienv.builtins__import()'
	builtins.__import__ = _import

	# Add MetaPathFinder
	## You can always get the original back via 'ienv.builtins__import()'
	sys.meta_path.insert(0, IEnvMetaPathFinder)


def free():
	"""Cease IEnv handling, affecting only **new** ienv-dependent imports.

	Nothing is deleted from `sys.modules`.
	As a result, if `import name` was IEnv-dependent, then:
	- Variables referring to an IEnv-dependent module will still work.
	- `sys.modules[ienv_prefix + 'name']` will still refer to the IEnv-dependent module.
	- Any stored IEnv-dependent `name`, ex. in a variable or a callback, will still refer to the IEnv-dependent module.

	There are a few gotchas (_Don't Do This_):
	- Dynamic ienv-dependent imports **will not work**.
	- `import _ienv_ienvname__name` will **only**  work if `sys.modules` still caches that name.
	"""
	global IENV_BASE_PATH, ALWAYS_PASSTHROUGH  # noqa: PLW0603
	IENV_BASE_PATH = None
	ALWAYS_PASSTHROUGH = None

	# Modify Builtins
	builtins.__import__ = builtins__import

	# Remove MetaPathFinder
	sys.meta_path.remove(IEnvMetaPathFinder)
