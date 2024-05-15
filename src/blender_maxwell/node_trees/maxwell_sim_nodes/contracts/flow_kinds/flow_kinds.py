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

	####################
	# - Class Methods
	####################
	@classmethod
	def scale_to_unit_system(
		cls,
		kind: typ.Self,
		flow_obj,
		unit_system: spux.UnitSystem,
	):
		# log.debug('%s: Scaling "%s" to Unit System', kind, str(flow_obj))
		## TODO: Use a hot-path logger.
		if kind == FlowKind.Value:
			return spux.scale_to_unit_system(
				flow_obj,
				unit_system,
			)
		if kind == FlowKind.LazyArrayRange:
			log.debug([kind, flow_obj, unit_system])
			return flow_obj.rescale_to_unit_system(unit_system)

		if kind == FlowKind.Params:
			return flow_obj.rescale_to_unit_system(unit_system)

		msg = 'Tried to scale unknown kind'
		raise ValueError(msg)

	####################
	# - Computed
	####################
	@property
	def flow_kind(self) -> str:
		return {
			FlowKind.Value: FlowKind.Value,
			FlowKind.Array: FlowKind.Array,
			FlowKind.LazyValueFunc: FlowKind.LazyValueFunc,
			FlowKind.LazyArrayRange: FlowKind.LazyArrayRange,
		}[self]

	@property
	def socket_shape(self) -> str:
		return {
			FlowKind.Value: 'CIRCLE',
			FlowKind.Array: 'SQUARE',
			FlowKind.LazyArrayRange: 'SQUARE',
			FlowKind.LazyValueFunc: 'DIAMOND',
		}[self]

	####################
	# - Blender Enum
	####################
	@staticmethod
	def to_name(v: typ.Self) -> str:
		return {
			FlowKind.Capabilities: 'Capabilities',
			FlowKind.Value: 'Value',
			FlowKind.Array: 'Array',
			FlowKind.LazyArrayRange: 'Range',
			FlowKind.LazyValueFunc: 'Lazy Value',
			FlowKind.Params: 'Parameters',
			FlowKind.Info: 'Information',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		return ''
