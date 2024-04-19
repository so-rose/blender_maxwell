"""Manages the registration of Blender classes, including delayed registrations that require access to Python dependencies.

Attributes:
	_ADDON_KEYMAP: Addon-specific keymap used to register operator hotkeys.
	DELAYED_REGISTRATIONS: Currently pending registration operations, which can be realized with `run_delayed_registration()`.

	REG__CLASSES: Currently registered Blender classes.
	_REGISTERED_HOTKEYS: Currently registered Blender keymap items.
"""

import enum
import typing as typ
from pathlib import Path

import bpy

from . import contracts as ct
from .nodeps.utils import simple_logger

log = simple_logger.get(__name__)

####################
# - Globals
####################
_REGISTERED_CLASSES: list[ct.BLClass] = []
_ADDON_KEYMAP: bpy.types.KeyMap | None = None
_REGISTERED_HOTKEYS: list[ct.BLKeymapItem] = []


####################
# - Delayed Registration
####################
class BLRegisterEvent(enum.StrEnum):
	DepsSatisfied = enum.auto()


DELAYED_REGISTRATIONS: dict[BLRegisterEvent, typ.Callable[[Path], None]] = {}


####################
# - Class Registration
####################
def register_classes(bl_register: list[ct.BLClass]) -> None:
	"""Registers a list of Blender classes.

	Parameters:
		bl_register: List of Blender classes to register.
	"""
	log.info('Registering %s Classes', len(bl_register))
	for cls in bl_register:
		if cls.bl_idname in _REGISTERED_CLASSES:
			msg = f'Skipping register of {cls.bl_idname}'
			log.info(msg)
			continue

		log.debug(
			'Registering Class %s',
			repr(cls),
		)
		bpy.utils.register_class(cls)
		_REGISTERED_CLASSES.append(cls)


def unregister_classes() -> None:
	"""Unregisters all previously registered Blender classes."""
	log.info('Unregistering %s Classes', len(_REGISTERED_CLASSES))
	for cls in reversed(_REGISTERED_CLASSES):
		log.debug(
			'Unregistering Class %s',
			repr(cls),
		)
		bpy.utils.unregister_class(cls)

	_REGISTERED_CLASSES.clear()


####################
# - Keymap Registration
####################
def register_hotkeys(hotkey_defs: list[dict]):
	"""Registers a list of Blender hotkey definitions.

	Parameters:
		hotkey_defs: List of Blender hotkey definitions to register.
	"""
	# Lazy-Load BL_NODE_KEYMAP
	global _ADDON_KEYMAP  # noqa: PLW0603
	if _ADDON_KEYMAP is None:
		_ADDON_KEYMAP = bpy.context.window_manager.keyconfigs.addon.keymaps.new(
			name=f'{ct.addon.NAME} Keymap',
		)
		log.info(
			'Registered Addon Keymap (Base for Keymap Items): %s',
			str(_ADDON_KEYMAP),
		)

	# Register Keymaps
	log.info('Registering %s Keymap Items', len(hotkey_defs))
	for keymap_item_def in hotkey_defs:
		keymap_item = _ADDON_KEYMAP.keymap_items.new(
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
		_REGISTERED_HOTKEYS.append(keymap_item)


def unregister_hotkeys():
	"""Unregisters all Blender hotkeys associated with the addon."""
	global _ADDON_KEYMAP  # noqa: PLW0603

	# Unregister Keymaps
	log.info('Unregistering %s Keymap Items', len(_REGISTERED_HOTKEYS))
	for keymap_item in reversed(_REGISTERED_HOTKEYS):
		log.debug(
			'Unregistered Keymap Item %s',
			repr(keymap_item),
		)
		_ADDON_KEYMAP.keymap_items.remove(keymap_item)

	# Lazy-Unload BL_NODE_KEYMAP
	if _ADDON_KEYMAP is not None:
		log.info(
			'Unregistered Keymap %s',
			repr(_ADDON_KEYMAP),
		)
		_REGISTERED_HOTKEYS.clear()
		_ADDON_KEYMAP = None


####################
# - Delayed Registration Semantics
####################
def delay_registration_until(
	delayed_reg_key: BLRegisterEvent,
	then_register_classes: typ.Callable[[Path], list[ct.BLClass]],
	then_register_hotkeys: typ.Callable[[Path], list[ct.KeymapItemDef]],
) -> None:
	"""Delays the registration of Blender classes that depend on certain Python dependencies, for which neither the location nor validity is yet known.

	The function that registers is stored in the module global `DELAYED_REGISTRATIONS`, indexed by `delayed_reg_key`.
	Once the PyDeps location and validity is determined, `run_delayed_registration()` can be used as a shorthand for accessing `DELAYED_REGISTRATIONS[delayed_reg_key]`.

	Parameters:
		delayed_reg_key: The identifier with which to index the registration callback.
		classes_cb: A function that takes a `sys.path`-compatible path to Python dependencies needed by the Blender classes in question, and returns a list of Blender classes to import.
			`register_classes()` will be used to actually register the returned Blender classes.
		hotkey_defs_cb: Similar, except for addon keymap items.

	Returns:
		A function that takes a `sys.path`-compatible path to the Python dependencies needed to import the given Blender classes.
	"""
	if delayed_reg_key in DELAYED_REGISTRATIONS:
		msg = f'Already delayed a registration with key {delayed_reg_key}'
		raise ValueError(msg)

	def register_cb(path_pydeps: Path):
		log.info(
			'Delayed Registration (key %s) with PyDeps Path: %s',
			delayed_reg_key,
			path_pydeps,
		)
		register_classes(then_register_classes(path_pydeps))
		register_hotkeys(then_register_hotkeys(path_pydeps))

	DELAYED_REGISTRATIONS[delayed_reg_key] = register_cb


def run_delayed_registration(
	delayed_reg_key: BLRegisterEvent, path_pydeps: Path
) -> None:
	"""Run a delayed registration, by using `delayed_reg_key` to lookup the correct path, passing `path_pydeps` to the registration.

	Parameters:
		delayed_reg_key: The identifier with which to index the registration callback.
			Must match the parameter with which the delayed registration was first declared.
		path_pydeps: The `sys.path`-compatible path to the Python dependencies that the classes need to have available in order to register.
	"""
	DELAYED_REGISTRATIONS.pop(delayed_reg_key)(path_pydeps)


def clear_delayed_registrations() -> None:
	"""Dequeue all queued delayed registrations."""
	DELAYED_REGISTRATIONS.clear()
