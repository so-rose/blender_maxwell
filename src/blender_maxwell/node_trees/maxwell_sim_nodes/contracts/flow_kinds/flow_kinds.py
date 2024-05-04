import enum
import typing as typ

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

log = logger.get(__name__)


class FlowKind(enum.StrEnum):
	"""Defines a kind of data that can flow between nodes.

	Each node link can be thought to contain **multiple pipelines for data to flow along**.
	Each pipeline is cached incrementally, and independently, of the others.
	Thus, the same socket can easily support several kinds of related data flow at the same time.

	Attributes:
		Capabilities: Describes a socket's linkeability with other sockets.
			Links between sockets with incompatible capabilities will be rejected.
			This doesn't need to be defined normally, as there is a default.
			However, in some cases, defining it manually to control linkeability more granularly may be desirable.
		Value: A generic object, which is "directly usable".
			This should be chosen when a more specific flow kind doesn't apply.
		Array: An object with dimensions, and possibly a unit.
			Whenever a `Value` is defined, a single-element `list` will also be generated by default as `Array`
			However, for any other array-like variants (or sockets that only represent array-like objects), `Array` should be defined manually.
		LazyValueFunc: A composable function.
			Can be used to represent computations for which all data is not yet known, or for which just-in-time compilation can drastically increase performance.
		LazyArrayRange: An object that generates an `Array` from range information (start/stop/step/spacing).
			This should be used instead of `Array` whenever possible.
		Param: A dictionary providing particular parameters for a lazy value.
		Info: An dictionary providing extra context about any aspect of flow.
	"""

	Capabilities = enum.auto()

	# Values
	Value = enum.auto()
	Array = enum.auto()

	# Lazy
	LazyValueFunc = enum.auto()
	LazyArrayRange = enum.auto()

	# Auxiliary
	Params = enum.auto()
	Info = enum.auto()

	@classmethod
	def scale_to_unit_system(
		cls,
		kind: typ.Self,
		value,
		unit_system: spux.UnitSystem,
	):
		if kind == cls.Value:
			return spux.scale_to_unit_system(
				value,
				unit_system,
			)
		if kind == cls.LazyArrayRange:
			return value.rescale_to_unit_system(unit_system)

		if kind == cls.Params:
			return value.rescale_to_unit_system(unit_system)

		msg = 'Tried to scale unknown kind'
		raise ValueError(msg)
