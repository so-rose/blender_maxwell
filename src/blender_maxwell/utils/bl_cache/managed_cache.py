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

"""Implements various key caches on instances of Blender objects, especially nodes and sockets."""

## TODO: Note that persist=True on cached_bl_property may cause a draw method to try and write to a Blender property, which Blender disallows.

import typing as typ

from blender_maxwell import contracts as ct
from blender_maxwell.utils import bl_instance, logger

from .signal import Signal

log = logger.get(__name__)


####################
# - Global Variables
####################
_CACHE_NONPERSIST: dict[bl_instance.InstanceID, dict[typ.Hashable, typ.Any]] = {}


####################
# - Create/Invalidate
####################
def bl_instance_nonpersist_cache(
	bl_instance: bl_instance.BLInstance,
) -> dict[typ.Hashable, typ.Any]:
	"""Retrieve the non-persistent cache of a BLInstance."""
	# Create Non-Persistent Cache Entry
	## Prefer explicit cache management to 'defaultdict'
	if _CACHE_NONPERSIST.get(bl_instance.instance_id) is None:
		_CACHE_NONPERSIST[bl_instance.instance_id] = {}

	return _CACHE_NONPERSIST[bl_instance.instance_id]


def invalidate_nonpersist_instance_id(instance_id: bl_instance.InstanceID) -> None:
	"""Invalidate any `instance_id` that might be utilizing cache space in `_CACHE_NONPERSIST`.

	Notes:
		This should be run by the `instance_id` owner in its `free()` method.

	Parameters:
		instance_id: The ID of the Blender object instance that's being freed.
	"""
	_CACHE_NONPERSIST.pop(instance_id, None)


####################
# - Access
####################
def read(
	bl_instance: bl_instance.BLInstance | None,
	key: typ.Hashable,
	use_nonpersist: bool = True,
	use_persist: bool = False,
) -> typ.Any | typ.Literal[Signal.CacheNotReady, Signal.CacheEmpty]:
	"""Read the cache associated with a Blender Instance, without writing to it.

	Attributes:
		key: The name to read from the instance-specific cache.
		use_nonpersist: If true, will always try the non-persistent cache first.
		use_persist: If true, will always try accessing the attribute `bl_instance,key`, where `key` is the value of the same-named parameter.
			Generally, such an attribute should be a `bpy.types.Property`.

	Return:
		The cache hit, if any; else `Signal.CacheEmpty`.
	"""
	# Check BLInstance Readiness
	if bl_instance is None:
		return Signal.CacheNotReady

	# Try Hit on Persistent Cache
	if use_persist:
		value = getattr(bl_instance, key, Signal.CacheEmpty)
		if value is not Signal.CacheEmpty:
			return value

	# Check if Instance ID is Available
	if not bl_instance.instance_id:
		log.debug(
			"Can't Get CachedBLProperty: Instance ID not (yet) defined on bl_instance.BLInstance %s",
			str(bl_instance),
		)
		return Signal.CacheNotReady

	# Try Hit on Non-Persistent Cache
	if use_nonpersist:
		cache_nonpersist = bl_instance_nonpersist_cache(bl_instance)
		value = cache_nonpersist.get(key, Signal.CacheEmpty)
		if value is not Signal.CacheEmpty:
			return value

	return Signal.CacheEmpty


def write(
	bl_instance: bl_instance.BLInstance,
	key: typ.Hashable,
	value: typ.Any,  ## TODO: "Serializable" type
	use_nonpersist: bool = True,
	use_persist: bool = False,
) -> None:
	"""Write to the cache associated with a Blender Instance.

	Attributes:
		key: The name to write to the instance-specific cache.
		use_nonpersist: If true, will always write to the non-persistent cache first.
		use_persist: If true, will always write to attribute `bl_instance.key`, where `key` is the value of the same-named parameter.
			Generally, such an attribute should be a `bpy.types.Property`.
		call_on_prop_changed: Whether to trigger `bl_instance.on_prop_changed()` with the
	"""
	# Check BLInstance Readiness
	if bl_instance is None:
		return

	# Try Write on Persistent Cache
	if use_persist:
		# log.critical('%s: Writing %s to %s.', str(bl_instance), str(value), str(key))
		setattr(bl_instance, key, value)

	if not bl_instance.instance_id:
		log.debug(
			"Can't Get CachedBLProperty: Instance ID not (yet) defined on bl_instance.BLInstance %s",
			str(bl_instance),
		)
		return

	# Try Write on Non-Persistent Cache
	if use_nonpersist:
		cache_nonpersist = bl_instance_nonpersist_cache(bl_instance)
		cache_nonpersist[key] = value


def invalidate_nonpersist(
	bl_instance: bl_instance.BLInstance,
	key: typ.Hashable,
) -> None:
	"""Invalidate a particular key of the non-persistent cache.

	**Persistent caches can't be invalidated without writing to them**.
	To get the same effect, consider using `write()` to write its default value (which must be manually tracked).
	"""
	# Check BLInstance Readiness
	if bl_instance is None:
		return
	if not bl_instance.instance_id:
		log.debug(
			"Can't Get CachedBLProperty: Instance ID not (yet) defined on bl_instance.BLInstance %s",
			str(bl_instance),
		)
		return

	# Retrieve Non-Persistent Cache
	cache_nonpersist = bl_instance_nonpersist_cache(bl_instance)
	cache_nonpersist.pop(key, None)
