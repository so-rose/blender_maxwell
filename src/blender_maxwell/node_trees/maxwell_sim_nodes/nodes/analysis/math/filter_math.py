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

"""Declares `FilterMathNode`."""

import enum
import typing as typ

import bpy
import jax.numpy as jnp

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class FilterOperation(enum.StrEnum):
	"""Valid operations for the `FilterMathNode`.

	Attributes:
		DimToVec: Shift last dimension to output.
		DimsToMat: Shift last 2 dimensions to output.
		PinLen1: Remove a len(1) dimension.
		Pin: Remove a len(n) dimension by selecting a particular index.
		Swap: Swap the positions of two dimensions.
	"""

	# Dimensions
	PinLen1 = enum.auto()
	Pin = enum.auto()
	Swap = enum.auto()

	# Interpret
	DimToVec = enum.auto()
	DimsToMat = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		FO = FilterOperation
		return {
			# Dimensions
			FO.PinLen1: 'pinₐ =1',
			FO.Pin: 'pinₐ ≈v',
			FO.Swap: 'a₁ ↔ a₂',
			# Interpret
			FO.DimToVec: '→ Vector',
			FO.DimsToMat: '→ Matrix',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return ''

	def are_dims_valid(self, dim_0: int | None, dim_1: int | None):
		return not (
			(
				dim_0 is None
				and self
				in [FilterOperation.PinLen1, FilterOperation.Pin, FilterOperation.Swap]
			)
			or (dim_1 is None and self == FilterOperation.Swap)
		)

	def jax_func(self, axis_0: int | None, axis_1: int | None):
		return {
			# Interpret
			FilterOperation.DimToVec: lambda data: data,
			FilterOperation.DimsToMat: lambda data: data,
			# Dimensions
			FilterOperation.PinLen1: lambda data: jnp.squeeze(data, axis_0),
			FilterOperation.Pin: lambda data, fixed_axis_idx: jnp.take(
				data, fixed_axis_idx, axis=axis_0
			),
			FilterOperation.Swap: lambda data: jnp.swapaxes(data, axis_0, axis_1),
		}[self]

	def transform_info(self, info: ct.InfoFlow, dim_0: str, dim_1: str):
		return {
			# Interpret
			FilterOperation.DimToVec: lambda: info.shift_last_input,
			FilterOperation.DimsToMat: lambda: info.shift_last_input.shift_last_input,
			# Dimensions
			FilterOperation.PinLen1: lambda: info.delete_dimension(dim_0),
			FilterOperation.Pin: lambda: info.delete_dimension(dim_0),
			FilterOperation.Swap: lambda: info.swap_dimensions(dim_0, dim_1),
		}[self]()


class FilterMathNode(base.MaxwellSimNode):
	r"""Applies a function that operates on the shape of the array.

	The shape, type, and interpretation of the input/output data is dynamically shown.

	# Socket Sets
	## Dimensions
	Alter the dimensions of the array.

	## Interpret
	Only alter the interpretation of the array data, which guides what it can be used for.

	These operations are **zero cost**, since the data itself is untouched.

	Attributes:
		operation: Operation to apply to the input.
	"""

	node_type = ct.NodeType.FilterMath
	bl_label = 'Filter Math'

	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Array),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Array),
	}

	####################
	# - Properties
	####################
	operation: FilterOperation = bl_cache.BLField(
		FilterOperation.PinLen1,
		prop_ui=True,
	)

	# Dimension Selection
	dim_0: enum.Enum = bl_cache.BLField(
		None, prop_ui=True, enum_cb=lambda self, _: self.search_dims()
	)
	dim_1: enum.Enum = bl_cache.BLField(
		None, prop_ui=True, enum_cb=lambda self, _: self.search_dims()
	)

	####################
	# - Computed
	####################
	@property
	def data_info(self) -> ct.InfoFlow | None:
		info = self._compute_input('Expr', kind=ct.FlowKind.Info)
		if not ct.FlowSignal.check(info):
			return info

		return None

	####################
	# - Search Dimensions
	####################
	def search_dims(self) -> list[ct.BLEnumElement]:
		if self.data_info is None:
			return []

		if self.operation == FilterOperation.PinLen1:
			dims = [
				(dim_name, dim_name, f'Dimension "{dim_name}" of length 1')
				for dim_name in self.data_info.dim_names
				if self.data_info.dim_lens[dim_name] == 1
			]
		elif self.operation in [FilterOperation.Pin, FilterOperation.Swap]:
			dims = [
				(dim_name, dim_name, f'Dimension "{dim_name}"')
				for dim_name in self.data_info.dim_names
			]
		else:
			return []

		return [(*dim, '', i) for i, dim in enumerate(dims)]

	####################
	# - UI
	####################
	def draw_label(self):
		FO = FilterOperation
		labels = {
			FO.PinLen1: lambda: f'Filter: Pin {self.dim_0} (len=1)',
			FO.Pin: lambda: f'Filter: Pin {self.dim_0}',
			FO.Swap: lambda: f'Filter: Swap {self.dim_0}|{self.dim_1}',
			FO.DimToVec: lambda: 'Filter: -> Vector',
			FO.DimsToMat: lambda: 'Filter: -> Matrix',
		}

		if (label := labels.get(self.operation)) is not None:
			return label()

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

		if self.operation in [FilterOperation.PinLen1, FilterOperation.Pin]:
			layout.prop(self, self.blfields['dim_0'], text='')

		if self.operation == FilterOperation.Swap:
			row = layout.row(align=True)
			row.prop(self, self.blfields['dim_0'], text='')
			row.prop(self, self.blfields['dim_1'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name='Expr',
		prop_name={'operation'},
		run_on_init=True,
	)
	def on_input_changed(self) -> None:
		self.dim_0 = bl_cache.Signal.ResetEnumItems
		self.dim_1 = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		# Trigger
		socket_name='Expr',
		prop_name={'dim_0', 'dim_1', 'operation'},
		run_on_init=True,
		# Loaded
		props={'operation', 'dim_0', 'dim_1'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def on_pin_changed(self, props: dict, input_sockets: dict):
		info = input_sockets['Expr']
		has_info = not ct.FlowSignal.check(info)
		if not has_info:
			return

		# "Dimensions"|"PIN": Add/Remove Input Socket
		if props['operation'] == FilterOperation.Pin and props['dim_0'] is not None:
			pinned_unit = info.dim_units[props['dim_0']]
			pinned_mathtype = info.dim_mathtypes[props['dim_0']]
			pinned_physical_type = spux.PhysicalType.from_unit(pinned_unit)

			wanted_mathtype = (
				spux.MathType.Complex
				if pinned_mathtype == spux.MathType.Complex
				and spux.MathType.Complex in pinned_physical_type.valid_mathtypes
				else spux.MathType.Real
			)

			# Get Current and Wanted Socket Defs
			current_bl_socket = self.loose_input_sockets.get('Value')

			# Determine Whether to Declare New Loose Input SOcket
			if (
				current_bl_socket is None
				or current_bl_socket.shape is not None
				or current_bl_socket.physical_type != pinned_physical_type
				or current_bl_socket.mathtype != wanted_mathtype
			):
				self.loose_input_sockets = {
					'Value': sockets.ExprSocketDef(
						active_kind=ct.FlowKind.Value,
						shape=None,
						physical_type=pinned_physical_type,
						mathtype=wanted_mathtype,
						default_unit=pinned_unit,
					),
				}
		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.LazyValueFunc,
		props={'operation', 'dim_0', 'dim_1'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Info}},
	)
	def compute_lazy_value_func(self, props: dict, input_sockets: dict):
		operation = props['operation']
		lazy_value_func = input_sockets['Expr'][ct.FlowKind.LazyValueFunc]
		info = input_sockets['Expr'][ct.FlowKind.Info]

		has_lazy_value_func = not ct.FlowSignal.check(lazy_value_func)
		has_info = not ct.FlowSignal.check(info)

		# Dimension(s)
		dim_0 = props['dim_0']
		dim_1 = props['dim_1']
		if (
			has_lazy_value_func
			and has_info
			and operation is not None
			and operation.are_dims_valid(dim_0, dim_1)
		):
			axis_0 = info.dim_names.index(dim_0) if dim_0 is not None else None
			axis_1 = info.dim_names.index(dim_1) if dim_1 is not None else None

			return lazy_value_func.compose_within(
				operation.jax_func(axis_0, axis_1),
				enclosing_func_args=[int] if operation == FilterOperation.Pin else [],
				supports_jax=True,
			)
		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Array,
		output_sockets={'Expr'},
		output_socket_kinds={
			'Expr': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Params},
		},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
	)
	def compute_array(self, output_sockets, unit_systems) -> ct.ArrayFlow:
		lazy_value_func = output_sockets['Expr'][ct.FlowKind.LazyValueFunc]
		params = output_sockets['Expr'][ct.FlowKind.Params]

		has_lazy_value_func = not ct.FlowSignal.check(lazy_value_func)
		has_params = not ct.FlowSignal.check(params)

		if has_lazy_value_func and has_params:
			unit_system = unit_systems['BlenderUnits']
			return ct.ArrayFlow(
				values=lazy_value_func.func_jax(
					*params.scaled_func_args(unit_system),
					**params.scaled_func_kwargs(unit_system),
				),
			)
		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary: Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		props={'dim_0', 'dim_1', 'operation'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def compute_info(self, props: dict, input_sockets: dict) -> ct.InfoFlow:
		operation = props['operation']
		info = input_sockets['Expr']

		has_info = not ct.FlowSignal.check(info)

		# Dimension(s)
		dim_0 = props['dim_0']
		dim_1 = props['dim_1']
		if (
			has_info
			and operation is not None
			and operation.are_dims_valid(dim_0, dim_1)
		):
			return operation.transform_info(info, dim_0, dim_1)

		return ct.FlowSignal.FlowPending

	####################
	# - Auxiliary: Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		props={'dim_0', 'dim_1', 'operation'},
		input_sockets={'Expr', 'Value'},
		input_socket_kinds={'Expr': {ct.FlowKind.Info, ct.FlowKind.Params}},
		input_sockets_optional={'Value': True},
	)
	def compute_params(self, props: dict, input_sockets: dict) -> ct.ParamsFlow:
		operation = props['operation']
		info = input_sockets['Expr'][ct.FlowKind.Info]
		params = input_sockets['Expr'][ct.FlowKind.Params]

		has_info = not ct.FlowSignal.check(info)
		has_params = not ct.FlowSignal.check(params)

		# Dimension(s)
		dim_0 = props['dim_0']
		dim_1 = props['dim_1']
		if (
			has_info
			and has_params
			and operation is not None
			and operation.are_dims_valid(dim_0, dim_1)
		):
			## Pinned Value
			pinned_value = input_sockets['Value']
			has_pinned_value = not ct.FlowSignal.check(pinned_value)

			if props['operation'] == FilterOperation.Pin and has_pinned_value:
				nearest_idx_to_value = info.dim_idx[dim_0].nearest_idx_of(
					pinned_value, require_sorted=True
				)

				return params.compose_within(enclosing_func_args=[nearest_idx_to_value])

			return params
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	FilterMathNode,
]
BL_NODES = {ct.NodeType.FilterMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
