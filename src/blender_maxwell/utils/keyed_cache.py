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

import functools
import inspect
import typing as typ

from blender_maxwell.utils import logger, serialize

log = logger.get(__name__)


class BLInstance(typ.Protocol):
	instance_id: str


class KeyedCache:
	def __init__(
		self,
		func: typ.Callable,
		exclude: set[str],
		encode: set[str],
	):
		# Function Information
		self.func: typ.Callable = func
		self.func_sig: inspect.Signature = inspect.signature(self.func)

		# Arg -> Key Information
		self.exclude: set[str] = exclude
		self.include: set[str] = set(self.func_sig.parameters.keys()) - exclude
		self.encode: set[str] = encode

		# Cache Information
		self.key_schema: tuple[str, ...] = tuple(
			[
				arg_name
				for arg_name in self.func_sig.parameters
				if arg_name not in exclude
			]
		)
		self.caches: dict[str | None, dict[tuple[typ.Any, ...], typ.Any]] = {}

	@property
	def is_method(self):
		return 'self' in self.exclude

	def cache(self, instance_id: str | None) -> dict[tuple[typ.Any, ...], typ.Any]:
		if self.caches.get(instance_id) is None:
			self.caches[instance_id] = {}

		return self.caches[instance_id]

	def _encode_key(self, arguments: dict[str, typ.Any]):
		## WARNING: Order of arguments matters. Arguments may contain 'exclude'd elements.
		return tuple(
			[
				(
					arg_value
					if arg_name not in self.encode
					else serialize.encode(arg_value)
				)
				for arg_name, arg_value in arguments.items()
				if arg_name in self.include
			]
		)

	def __get__(
		self,
		bl_instance: BLInstance | None,
		owner: type[BLInstance],
	) -> typ.Callable:
		_func = functools.partial(self, bl_instance)
		_func.invalidate = functools.partial(
			self.__class__.invalidate, self, bl_instance
		)
		return _func

	def __call__(self, *args, **kwargs):
		# Test Argument Bindability to Decorated Function
		try:
			bound_args = self.func_sig.bind(*args, **kwargs)
		except TypeError as ex:
			msg = f'Can\'t bind arguments (args={args}, kwargs={kwargs}) to @keyed_cache-decorated function "{self.func.__name__}" (signature: {self.func_sig})"'
			raise ValueError(msg) from ex

		# Check that Parameters for Keying the Cache are Available
		bound_args.apply_defaults()
		all_arg_keys = set(bound_args.arguments.keys())
		if not self.include <= (all_arg_keys - self.exclude):
			msg = f'Arguments spanning the keyed cached ({self.include}) are not available in the non-excluded arguments passed to "{self.func.__name__}": {all_arg_keys - self.exclude}'
			raise ValueError(msg)

		# Create Keyed Cache Entry
		key = self._encode_key(bound_args.arguments)
		cache = self.cache(args[0].instance_id if self.is_method else None)
		if (value := cache.get(key)) is None:
			value = self.func(*args, **kwargs)
			cache[key] = value

		return value

	def invalidate(
		self,
		bl_instance: BLInstance | None,
		**arguments: dict[str, typ.Any],
	) -> dict[str, typ.Any]:
		# Determine Wildcard Arguments
		wildcard_arguments = {
			arg_name for arg_name, arg_value in arguments.items() if arg_value is ...
		}

		# Compute Keys to Invalidate
		arguments_hashable = {
			arg_name: serialize.encode(arg_value)
			if arg_name in self.encode and arg_name not in wildcard_arguments
			else arg_value
			for arg_name, arg_value in arguments.items()
		}
		cache = self.cache(bl_instance.instance_id if self.is_method else None)
		for key in list(cache.keys()):
			if all(
				arguments_hashable.get(arg_name) == arg_value
				for arg_name, arg_value in zip(self.key_schema, key, strict=True)
				if arg_name not in wildcard_arguments
			):
				cache.pop(key)


def keyed_cache(exclude: set[str], encode: set[str] = frozenset()) -> typ.Callable:
	def decorator(func: typ.Callable) -> typ.Callable:
		return KeyedCache(
			func,
			exclude=exclude,
			encode=encode,
		)

	return decorator
