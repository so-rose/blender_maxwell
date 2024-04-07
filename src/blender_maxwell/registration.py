"""Manages the registration of Blender classes, including delayed registrations that require access to Python dependencies.

Attributes:
	BL_KEYMAP: Addon-specific keymap used to register operator hotkeys. REG__CLASSES: Currently registered Blender classes.
	REG__KEYMAP_ITEMS: Currently registered Blender keymap items.
	DELAYED_REGISTRATIONS: Currently pending registration operations, which can be realized with `run_delayed_registration()`.
	EVENT__DEPS_SATISFIED: A constant representing a semantic choice of key for `DELAYED_REGISTRATIONS`.
"""

import typing as typ
from pathlib import Path

import bpy

from .nodeps.utils import simple_logger

log = simple_logger.get(__name__)

# TODO: More types for these things!
DelayedRegKey: typ.TypeAlias = str
BLClass: typ.TypeAlias = (
	bpy.types.Panel
	| bpy.types.UIList
	| bpy.types.Menu
	| bpy.types.Header
	| bpy.types.Operator
	| bpy.types.KeyingSetInfo
	| bpy.types.RenderEngine
	| bpy.types.AssetShelf
	| bpy.types.FileHandler
)
BLKeymapItem: typ.TypeAlias = typ.Any  ## TODO: Better Type
KeymapItemDef: typ.TypeAlias = typ.Any  ## TODO: Better Type

####################
# - Globals
####################
BL_KEYMAP: bpy.types.KeyMap | None = None

REG__CLASSES: list[BLClass] = []
REG__KEYMAP_ITEMS: list[BLKeymapItem] = []

DELAYED_REGISTRATIONS: dict[DelayedRegKey, typ.Callable[[Path], None]] = {}

####################
# - Delayed Registration Keys
####################
EVENT__DEPS_SATISFIED: DelayedRegKey = 'on_deps_satisfied'


####################
# - Class Registration
####################
def register_classes(bl_register: list[BLClass]) -> None:
	"""Registers a Blender class, allowing it to hook into relevant Blender features.

	Caches registered classes in the module global `REG__CLASSES`.

	Parameters:
		bl_register: List of Blender classes to register.
	"""
	log.info('Registering %s Classes', len(bl_register))
	for cls in bl_register:
		if cls.bl_idname in REG__CLASSES:
			msg = f'Skipping register of {cls.bl_idname}'
			log.info(msg)
			continue

		log.debug(
			'Registering Class %s',
			repr(cls),
		)
		bpy.utils.register_class(cls)
		REG__CLASSES.append(cls)


def unregister_classes() -> None:
	"""Unregisters all previously registered Blender classes.

	All previously registered Blender classes can be found in the module global variable `REG__CLASSES`.
	"""
	log.info('Unregistering %s Classes', len(REG__CLASSES))
	for cls in reversed(REG__CLASSES):
		log.debug(
			'Unregistering Class %s',
			repr(cls),
		)
		bpy.utils.unregister_class(cls)

	REG__CLASSES.clear()


####################
# - Keymap Registration
####################
def register_keymap_items(keymap_item_defs: list[dict]):
	# Lazy-Load BL_NODE_KEYMAP
	global BL_KEYMAP  # noqa: PLW0603
	if BL_KEYMAP is None:
		BL_KEYMAP = bpy.context.window_manager.keyconfigs.addon.keymaps.new(
			name='Node Editor',
			space_type='NODE_EDITOR',
		)
		log.info(
			'Registered Keymap %s',
			str(BL_KEYMAP),
		)

	# Register Keymaps
	log.info('Registering %s Keymap Items', len(keymap_item_defs))
	for keymap_item_def in keymap_item_defs:
		keymap_item = BL_KEYMAP.keymap_items.new(
			*keymap_item_def['_'],
			ctrl=keymap_item_def['ctrl'],
			shift=keymap_item_def['shift'],
			alt=keymap_item_def['alt'],
		)
		log.debug(
			'Registered Keymap Item %s with spec %s',
			repr(keymap_item),
			keymap_item_def,
		)
		REG__KEYMAP_ITEMS.append(keymap_item)


def unregister_keymap_items():
	global BL_KEYMAP  # noqa: PLW0603

	# Unregister Keymaps
	log.info('Unregistering %s Keymap Items', len(REG__KEYMAP_ITEMS))
	for keymap_item in reversed(REG__KEYMAP_ITEMS):
		log.debug(
			'Unregistered Keymap Item %s',
			repr(keymap_item),
		)
		BL_KEYMAP.keymap_items.remove(keymap_item)

	# Lazy-Unload BL_NODE_KEYMAP
	if BL_KEYMAP is not None:
		log.info(
			'Unregistered Keymap %s',
			repr(BL_KEYMAP),
		)
		REG__KEYMAP_ITEMS.clear()
		BL_KEYMAP = None


####################
# - Delayed Registration Semantics
####################
def delay_registration(
	delayed_reg_key: DelayedRegKey,
	classes_cb: typ.Callable[[Path], list[BLClass]],
	keymap_item_defs_cb: typ.Callable[[Path], list[KeymapItemDef]],
) -> None:
	"""Delays the registration of Blender classes that depend on certain Python dependencies, for which neither the location nor validity is yet known.

	The function that registers is stored in the module global `DELAYED_REGISTRATIONS`, indexed by `delayed_reg_key`.
	Once the PyDeps location and validity is determined, `run_delayed_registration()` can be used as a shorthand for accessing `DELAYED_REGISTRATIONS[delayed_reg_key]`.

	Parameters:
		delayed_reg_key: The identifier with which to index the registration callback.
			Module-level constants like `EVENT__DEPS_SATISFIED` are a good choice.
		classes_cb: A function that takes a `sys.path`-compatible path to Python dependencies needed by the Blender classes in question, and returns a list of Blender classes to import.
			`register_classes()` will be used to actually register the returned Blender classes.
		keymap_item_defs_cb: Similar, except for addon keymap items.

	Returns:
		A function that takes a `sys.path`-compatible path to the Python dependencies needed to import the given Blender classes.
	"""
	if delayed_reg_key in DELAYED_REGISTRATIONS:
		msg = f'Already delayed a registration with key {delayed_reg_key}'
		raise ValueError(msg)

	def register_cb(path_pydeps: Path):
		log.info(
			'Running Delayed Registration (key %s) with PyDeps: %s',
			delayed_reg_key,
			path_pydeps,
		)
		register_classes(classes_cb(path_pydeps))
		register_keymap_items(keymap_item_defs_cb(path_pydeps))

	DELAYED_REGISTRATIONS[delayed_reg_key] = register_cb


def run_delayed_registration(delayed_reg_key: DelayedRegKey, path_pydeps: Path) -> None:
	"""Run a delayed registration, by using `delayed_reg_key` to lookup the correct path, passing `path_pydeps` to the registration.

	Parameters:
		delayed_reg_key: The identifier with which to index the registration callback.
			Must match the parameter with which the delayed registration was first declared.
		path_pydeps: The `sys.path`-compatible path to the Python dependencies that the classes need to have available in order to register.
	"""
	register_cb = DELAYED_REGISTRATIONS.pop(delayed_reg_key)
	register_cb(path_pydeps)
