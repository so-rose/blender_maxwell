import typing as typ
from pathlib import Path

import bpy

from .utils import logger as _logger

log = _logger.get()

# TODO: More types for these things!
DelayedRegKey: typ.TypeAlias = str
BLClass: typ.TypeAlias = typ.Any  ## TODO: Better Type
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
# - Constants
####################
EVENT__DEPS_SATISFIED: str = 'on_deps_satisfied'


####################
# - Class Registration
####################
def register_classes(bl_register: list):
	for cls in bl_register:
		if cls.bl_idname in REG__CLASSES:
			msg = f'Skipping register of {cls.bl_idname}'
			log.info(msg)
			continue

		bpy.utils.register_class(cls)
		REG__CLASSES.append(cls)


def unregister_classes():
	for cls in reversed(REG__CLASSES):
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

	# Register Keymaps
	for keymap_item_def in keymap_item_defs:
		keymap_item = BL_KEYMAP.keymap_items.new(
			*keymap_item_def['_'],
			ctrl=keymap_item_def['ctrl'],
			shift=keymap_item_def['shift'],
			alt=keymap_item_def['alt'],
		)
		REG__KEYMAP_ITEMS.append(keymap_item)
	


def unregister_keymap_items():
	global BL_KEYMAP  # noqa: PLW0603

	# Unregister Keymaps
	for keymap_item in reversed(REG__KEYMAP_ITEMS):
		BL_KEYMAP.keymap_items.remove(keymap_item)

	# Lazy-Unload BL_NODE_KEYMAP
	if BL_KEYMAP is not None:
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
	if delayed_reg_key in DELAYED_REGISTRATIONS:
		msg = f'Already delayed a registration with key {delayed_reg_key}'
		raise ValueError(msg)

	def register_cb(path_deps: Path):
		register_classes(classes_cb(path_deps))
		register_keymap_items(keymap_item_defs_cb(path_deps))

	DELAYED_REGISTRATIONS[delayed_reg_key] = register_cb


def run_delayed_registration(
	delayed_reg_key: DelayedRegKey, path_deps: Path
) -> None:
	register_cb = DELAYED_REGISTRATIONS.pop(delayed_reg_key)
	register_cb(path_deps)
