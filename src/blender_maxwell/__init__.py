"""A Blender-based system for electromagnetic simulation design and analysis, with deep Tidy3D integration.

# `bl_info`
`bl_info` declares information about the addon to Blender.

However, it is not _dynamically_ read: Blender traverses it using `ast.parse`.
This makes it difficult to synchronize `bl_info` with the project's `pyproject.toml`.
As a workaround, **the addon zip-packer will replace `bl_info` entries**.

The following `bl_info` entries are currently replaced when the ZIP is built:

- `description`: To match the description in `pyproject.toml`.
- `version`: To match the version in `pyproject.toml`.

For more information, see `scripts.pack.BL_INFO_REPLACEMENTS`.

**NOTE**: The find/replace procedure is "dumb" (aka. no regex, no `ast` traversal, etc.).

This is surprisingly robust, so long as use of the deterministic code-formatter `ruff fmt` is enforced.

Still. Be careful around `bl_info`.

Attributes:
	bl_info: Information about the addon declared to Blender.
	BL_REGISTER_BEFORE_DEPS: Blender classes to register before dependencies are verified as installed.
	BL_HOTKEYS: Blender keymap item defs to register before dependencies are verified as installed.
"""

from pathlib import Path

from .nodeps.utils import simple_logger

# Initialize Logging Defaults
## Initial logger settings (ex. log level) must be set somehow.
## The Addon ZIP-packer makes this decision, and packs it into files.
## AddonPreferences will, once loaded, override this.
_PATH_ADDON_ROOT = Path(__file__).resolve().parent
_PATH_BOOTSTRAP_LOG_LEVEL = _PATH_ADDON_ROOT / '.bootstrap_log_level'
with _PATH_BOOTSTRAP_LOG_LEVEL.open('r') as f:
	_BOOTSTRAP_LOG_LEVEL = int(f.read().strip())

simple_logger.init_simple_logger_defaults(console_level=_BOOTSTRAP_LOG_LEVEL)

# Import Statements
import bpy  # noqa: E402

from . import contracts as ct  # noqa: E402
from . import preferences, registration  # noqa: E402
from .nodeps import operators as nodeps_operators  # noqa: E402
from .nodeps.utils import pydeps  # noqa: E402

log = simple_logger.get(__name__)

####################
# - Addon Information
####################
bl_info = {
	'name': 'Maxwell PDE Sim and Viz',
	'blender': (4, 1, 0),
	'category': 'Node',
	'description': 'Placeholder',
	'author': 'Sofus Albert Høgsbro Rose',
	'version': (0, 0, 0),
	'wiki_url': 'https://git.sofus.io/dtu-courses/bsc_thesis',
	'tracker_url': 'https://git.sofus.io/dtu-courses/bsc_thesis/issues',
}


####################
# - Load and Register Addon
####################
BL_REGISTER_BEFORE_DEPS: list[ct.BLClass] = [
	*nodeps_operators.BL_REGISTER,
	*preferences.BL_REGISTER,
]

## TODO: BL_HANDLERS and BL_SOCKET_DEFS

BL_HOTKEYS_BEFORE_DEPS: list[ct.KeymapItemDef] = [
	*nodeps_operators.BL_HOTKEYS,
]


def load_main_blclasses(path_pydeps: Path) -> list[ct.BLClass]:
	"""Imports all addon classes that rely on Python dependencies.

	Notes:
		`sys.path` is modified while executing this function.

	Parameters:
		path_pydeps: The path to the Python dependencies.

	Returns:
		An ordered list of Blender classes to register.
	"""
	with pydeps.importable_addon_deps(path_pydeps):
		from . import assets, node_trees, operators
	return [
		*operators.BL_REGISTER,
		*assets.BL_REGISTER,
		*node_trees.BL_REGISTER,
	]


def load_main_blhotkeys(path_deps: Path) -> list[ct.KeymapItemDef]:
	"""Imports all keymap item defs that rely on Python dependencies.

	Notes:
		`sys.path` is modified while executing this function.

	Parameters:
		path_pydeps: The path to the Python dependencies.

	Returns:
		An ordered list of Blender keymap item defs to register.
	"""
	with pydeps.importable_addon_deps(path_deps):
		from . import assets, operators
	return [
		*operators.BL_HOTKEYS,
		*assets.BL_HOTKEYS,
	]


####################
# - Registration
####################
@bpy.app.handlers.persistent
def manage_pydeps(*_):
	# ct.addon.operator(
	# ct.OperatorType.ManagePyDeps,
	# 'INVOKE_DEFAULT',
	# path_addon_pydeps='',
	# path_addon_reqs='',
	# )
	ct.addon.prefs().on_addon_pydeps_changed(show_popup_if_deps_invalid=True)


def register() -> None:
	"""Implements a multi-stage addon registration, which accounts for Python dependency management.

	# Multi-Stage Registration
	The trouble is that many classes in our addon might require Python dependencies.

	## Stage 1: Barebones Addon
	Many classes in our addon might require Python dependencies.
	However, they may not yet be installed.

	To solve this bootstrapping problem in a streamlined manner, we only **guarantee** the registration of a few key classes, including:

	- `AddonPreferences`: The addon preferences provide an interface for the user to fix Python dependency problems, thereby triggering subsequent stages.
	- `InstallPyDeps`: An operator that installs missing Python dependencies, using Blender's embeded `pip`.
	- `UninstallPyDeps`: An operator that uninstalls Python dependencies.

	**These classes provide just enough interface to help the user install the missing Python dependencies**.

	## Stage 2: Declare Delayed Registration
	We may not be able to register any classes that rely on Python dependencies.
	However, we can use `registration.delay_registration()` to **delay the registration until it is determined that the Python dependencies are satisfied**.`

	For now, we just pass a callback that will import + return a list of classes to register (`load_main_blclasses()`) when the time comes.

	## Stage 3: Trigger "PyDeps Changed"
	The addon preferences is responsible for storing (and exposing to the user) the path to the Python dependencies.

	Thus, the addon preferences method `on_addon_pydeps_changed()` has the responsibility for checking when the dependencies are valid, and running the delayed registrations (and any other delayed setup) in response.
	In general, `on_addon_pydeps_changed()` runs whenever the PyDeps path is changed, but it can also be run manually.

	As the last part of this process, that's exactly what `register()` does: Runs `on_addon_pydeps_changed()` manually.
	Depending on the addon preferences (which persist), one of two things can happen:

	1. **Deps Satisfied**: The addon will load without issue: The just-declared "delayed registrations" will run immediately, and all is well.
	2. **Deps Not Satisfied**: The user must take action to fix the conflicts due to Python dependencies, before the addon can load. **A popup will show to help the user do so.


	Notes:
		Called by Blender when enabling the addon.
	"""
	log.info('Commencing Registration of Addon: %s', ct.addon.NAME)
	bpy.app.handlers.load_post.append(manage_pydeps)

	# Register Barebones Addon
	## Contains all no-dependency BLClasses:
	## - Contains AddonPreferences.
	## Contains all BLClasses from 'nodeps'.
	registration.register_classes(BL_REGISTER_BEFORE_DEPS)
	registration.register_hotkeys(BL_HOTKEYS_BEFORE_DEPS)

	# Delay Complete Registration until DEPS_SATISFIED
	registration.delay_registration_until(
		registration.BLRegisterEvent.DepsSatisfied,
		then_register_classes=load_main_blclasses,
		then_register_hotkeys=load_main_blhotkeys,
	)

	# Trigger PyDeps Check
	## Deps ARE OK: Delayed registration will trigger.
	## Deps NOT OK: User must fix the pydeps, then trigger this method.
	ct.addon.prefs().on_addon_pydeps_changed()


def unregister() -> None:
	"""Unregisters anything that was registered by the addon.

	Notes:
		Run by Blender when disabling the addon.

		This doesn't clean `sys.modules`.
		To fully revert to Blender's state before the addon was in use (especially various import-related caches in the Python process), Blender must be restarted.
	"""
	log.info('Starting %s Unregister', ct.addon.NAME)
	registration.unregister_classes()
	registration.unregister_hotkeys()
	registration.clear_delayed_registrations()
	log.info('Finished %s Unregister', ct.addon.NAME)
