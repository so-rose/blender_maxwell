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
import jax.lax as jlax
import jax.numpy as jnp
import sympy as sp

from blender_maxwell.utils import bl_cache, logger, sim_symbols
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

	# Slice
	SliceIdx = enum.auto()

	# Pin
	PinLen1 = enum.auto()
	Pin = enum.auto()
	PinIdx = enum.auto()

	# Reinterpret
	Swap = enum.auto()
	SetDim = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		FO = FilterOperation
		return {
			# Slice
			FO.SliceIdx: 'a[...]',
			# Pin
			FO.PinLen1: 'pinₐ =1',
			FO.Pin: 'pinₐ ≈v',
			FO.PinIdx: 'pinₐ =a[v]',
			# Reinterpret
			FO.Swap: 'a₁ ↔ a₂',
			FO.SetDim: 'setₐ =v',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		FO = FilterOperation
		return (
			str(self),
			FO.to_name(self),
			FO.to_name(self),
			FO.to_icon(self),
			i,
		)

	####################
	# - Ops from Info
	####################
	@staticmethod
	def by_info(info: ct.InfoFlow) -> list[typ.Self]:
		FO = FilterOperation
		operations = []

		# Slice
		if info.dim_names:
			operations.append(FO.SliceIdx)

		# Pin
		## PinLen1
		## -> There must be a dimension with length 1.
		if 1 in list(info.dim_lens.values()):
			operations.append(FO.PinLen1)

		## Pin | PinIdx
		## -> There must be a dimension, full stop.
		if info.dim_names:
			operations += [FO.Pin, FO.PinIdx]

		# Reinterpret
		## Swap
		## -> There must be at least two dimensions.
		if len(info.dim_names) >= 2:  # noqa: PLR2004
			operations.append(FO.Swap)

		## SetDim
		## -> There must be a dimension to correct.
		if info.dim_names:
			operations.append(FO.SetDim)

		return operations

	####################
	# - Computed Properties
	####################
	@property
	def func_args(self) -> list[spux.MathType]:
		FO = FilterOperation
		return {
			# Pin
			FO.Pin: [spux.MathType.Integer],
			FO.PinIdx: [spux.MathType.Integer],
		}.get(self, [])

	####################
	# - Methods
	####################
	@property
	def num_dim_inputs(self) -> None:
		FO = FilterOperation
		return {
			# Slice
			FO.SliceIdx: 1,
			# Pin
			FO.PinLen1: 1,
			FO.Pin: 1,
			FO.PinIdx: 1,
			# Reinterpret
			FO.Swap: 2,
			FO.SetDim: 1,
		}[self]

	def valid_dims(self, info: ct.InfoFlow) -> list[typ.Self]:
		FO = FilterOperation
		match self:
			case FO.SliceIdx:
				return info.dim_names

			# PinLen1: Only allow dimensions with length=1.
			case FO.PinLen1:
				return [
					dim_name
					for dim_name in info.dim_names
					if info.dim_lens[dim_name] == 1
				]

			# Pin: Only allow dimensions with known indexing.
			case FO.Pin:
				return [
					dim_name
					for dim_name in info.dim_names
					if info.dim_has_coords[dim_name] != 0
				]

			case FO.PinIdx | FO.Swap:
				return info.dim_names

			case FO.SetDim:
				return [
					dim_name
					for dim_name in info.dim_names
					if info.dim_mathtypes[dim_name] == spux.MathType.Integer
				]

		return []

	def are_dims_valid(
		self, info: ct.InfoFlow, dim_0: str | None, dim_1: str | None
	) -> bool:
		"""Check whether the given dimension inputs are valid in the context of this operation, and of the information."""
		return (self.num_dim_inputs in [1, 2] and dim_0 in self.valid_dims(info)) or (
			self.num_dim_inputs == 2 and dim_1 in self.valid_dims(info)
		)

	####################
	# - UI
	####################
	def jax_func(
		self,
		axis_0: int | None,
		axis_1: int | None,
		slice_tuple: tuple[int, int, int] | None = None,
	):
		FO = FilterOperation
		return {
			# Pin
			FO.SliceIdx: lambda expr: jlax.slice_in_dim(
				expr, slice_tuple[0], slice_tuple[1], slice_tuple[2], axis=axis_0
			),
			# Pin
			FO.PinLen1: lambda expr: jnp.squeeze(expr, axis_0),
			FO.Pin: lambda expr, idx: jnp.take(expr, idx, axis=axis_0),
			FO.PinIdx: lambda expr, idx: jnp.take(expr, idx, axis=axis_0),
			# Reinterpret
			FO.Swap: lambda expr: jnp.swapaxes(expr, axis_0, axis_1),
			FO.SetDim: lambda expr: expr,
		}[self]

	def transform_info(
		self,
		info: ct.InfoFlow,
		dim_0: str,
		dim_1: str,
		slice_tuple: tuple[int, int, int] | None = None,
		corrected_dim: tuple[str, tuple[str, ct.ArrayFlow | ct.LazyArrayRangeFlow]]
		| None = None,
	):
		FO = FilterOperation
		return {
			FO.SliceIdx: lambda: info.slice_dim(dim_0, slice_tuple),
			# Pin
			FO.PinLen1: lambda: info.delete_dimension(dim_0),
			FO.Pin: lambda: info.delete_dimension(dim_0),
			FO.PinIdx: lambda: info.delete_dimension(dim_0),
			# Reinterpret
			FO.Swap: lambda: info.swap_dimensions(dim_0, dim_1),
			FO.SetDim: lambda: info.replace_dim(*corrected_dim),
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
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.LazyValueFunc),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.LazyValueFunc),
	}

	####################
	# - Properties: Expr InfoFlow
	####################
	@events.on_value_changed(
		socket_name={'Expr'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
		input_sockets_optional={'Expr': True},
	)
	def on_input_exprs_changed(self, input_sockets) -> None:  # noqa: D102
		has_info = not ct.FlowSignal.check(input_sockets['Expr'])

		info_pending = ct.FlowSignal.check_single(
			input_sockets['Expr'], ct.FlowSignal.FlowPending
		)

		if has_info and not info_pending:
			self.expr_info = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def expr_info(self) -> ct.InfoFlow | None:
		info = self._compute_input('Expr', kind=ct.FlowKind.Info, optional=True)
		has_info = not ct.FlowSignal.check(info)
		if has_info:
			return info

		return None

	####################
	# - Properties: Operation
	####################
	operation: FilterOperation = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_info'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.expr_info is not None:
			return [
				operation.bl_enum_element(i)
				for i, operation in enumerate(FilterOperation.by_info(self.expr_info))
			]
		return []

	####################
	# - Properties: Dimension Selection
	####################
	dim_0: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_dims(),
		cb_depends_on={'operation', 'expr_info'},
	)
	dim_1: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_dims(),
		cb_depends_on={'operation', 'expr_info'},
	)

	def search_dims(self) -> list[ct.BLEnumElement]:
		if self.expr_info is not None and self.operation is not None:
			return [
				(dim_name, dim_name, dim_name, '', i)
				for i, dim_name in enumerate(self.operation.valid_dims(self.expr_info))
			]
		return []

	####################
	# - Properties: Slice
	####################
	slice_tuple: tuple[int, int, int] = bl_cache.BLField([0, 1, 1])

	####################
	# - Properties: Unit
	####################
	set_dim_symbol: sim_symbols.CommonSimSymbol = bl_cache.BLField(
		sim_symbols.CommonSimSymbol.X
	)

	set_dim_active_unit: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_valid_units(),
		cb_depends_on={'set_dim_symbol'},
	)

	def search_valid_units(self) -> list[ct.BLEnumElement]:
		"""Compute Blender enum elements of valid units for the current `physical_type`."""
		physical_type = self.set_dim_symbol.sim_symbol.physical_type
		if physical_type is not spux.PhysicalType.NonPhysical:
			return [
				(sp.sstr(unit), spux.sp_to_str(unit), sp.sstr(unit), '', i)
				for i, unit in enumerate(physical_type.valid_units)
			]
		return []

	@bl_cache.cached_bl_property(depends_on={'set_dim_active_unit'})
	def set_dim_unit(self) -> spux.Unit | None:
		if self.set_dim_active_unit is not None:
			return spux.unit_str_to_unit(self.set_dim_active_unit)

		return None

	####################
	# - UI
	####################
	def draw_label(self):
		FO = FilterOperation
		match self.operation:
			# Slice
			case FO.SliceIdx:
				slice_str = ':'.join([str(v) for v in self.slice_tuple])
				return f'Filter: {self.dim_0}[{slice_str}]'

			# Pin
			case FO.PinLen1:
				return f'Filter: Pin {self.dim_0}[0]'
			case FO.Pin:
				return f'Filter: Pin {self.dim_0}[...]'
			case FO.PinIdx:
				pin_idx_axis = self._compute_input(
					'Axis', kind=ct.FlowKind.Value, optional=True
				)
				has_pin_idx_axis = not ct.FlowSignal.check(pin_idx_axis)
				if has_pin_idx_axis:
					return f'Filter: Pin {self.dim_0}[{pin_idx_axis}]'
				return self.bl_label

			# Reinterpret
			case FO.Swap:
				return f'Filter: Swap [{self.dim_0}]|[{self.dim_1}]'
			case FO.SetDim:
				return f'Filter: Set [{self.dim_0}]'

			case _:
				return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

		if self.operation is not None:
			match self.operation.num_dim_inputs:
				case 1:
					layout.prop(self, self.blfields['dim_0'], text='')
				case 2:
					row = layout.row(align=True)
					row.prop(self, self.blfields['dim_0'], text='')
					row.prop(self, self.blfields['dim_1'], text='')

			if self.operation is FilterOperation.SliceIdx:
				layout.prop(self, self.blfields['slice_tuple'], text='')

			if self.operation is FilterOperation.SetDim:
				row = layout.row(align=True)
				row.prop(self, self.blfields['set_dim_symbol'], text='')
				row.prop(self, self.blfields['set_dim_active_unit'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name='Expr',
		prop_name={'operation', 'dim_0', 'dim_1'},
		# Loaded
		props={'operation', 'dim_0', 'dim_1'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def on_pin_factors_changed(self, props: dict, input_sockets: dict):
		"""Synchronize loose input sockets to match the dimension-pinning method declared in `self.operation`.

		To "pin" an axis, a particular index must be chosen to "extract".
		One might choose axes of length 1 ("squeeze"), choose a particular index, or choose a coordinate that maps to a particular index.

		Those last two options requires more information from the user: Which index?
		Which coordinate?
		To answer these questions, we create an appropriate loose input socket containing this data, so the user can make their decision.
		"""
		info = input_sockets['Expr']
		has_info = not ct.FlowSignal.check(info)
		if not has_info:
			return

		# Pin Dim by-Value: Synchronize Input Socket
		## -> The user will be given a socket w/correct mathtype, unit, etc. .
		## -> Internally, this value will map to a particular index.
		if props['operation'] is FilterOperation.Pin and props['dim_0'] is not None:
			# Deduce Pinned Information
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
			## -> 'Value' may already exist. If not, all is well.
			current_bl_socket = self.loose_input_sockets.get('Value')

			# Determine Whether to Construct
			## -> If nothing needs to change, then nothing changes.
			if (
				current_bl_socket is None
				or current_bl_socket.size is not spux.NumberSize1D.Scalar
				or current_bl_socket.physical_type != pinned_physical_type
				or current_bl_socket.mathtype != wanted_mathtype
			):
				self.loose_input_sockets = {
					'Value': sockets.ExprSocketDef(
						active_kind=ct.FlowKind.Value,
						physical_type=pinned_physical_type,
						mathtype=wanted_mathtype,
						default_unit=pinned_unit,
					),
				}

		# Pin Dim by-Index: Synchronize Input Socket
		## -> The user will be given a simple integer socket.
		elif (
			props['operation'] is FilterOperation.PinIdx and props['dim_0'] is not None
		):
			current_bl_socket = self.loose_input_sockets.get('Axis')
			if (
				current_bl_socket is None
				or current_bl_socket.size is not spux.NumberSize1D.Scalar
				or current_bl_socket.physical_type != spux.PhysicalType.NonPhysical
				or current_bl_socket.mathtype != spux.MathType.Integer
			):
				self.loose_input_sockets = {
					'Axis': sockets.ExprSocketDef(
						active_kind=ct.FlowKind.Value,
						mathtype=spux.MathType.Integer,
					)
				}

		# Set Dim: Synchronize Input Socket
		## -> The user must provide a (ℤ) -> ℝ array.
		## -> It must be of identical length to the replaced axis.
		elif (
			props['operation'] is FilterOperation.SetDim
			and props['dim_0'] is not None
			and info.dim_mathtypes[props['dim_0']] is spux.MathType.Integer
			and info.dim_physical_types[props['dim_0']] is spux.PhysicalType.NonPhysical
		):
			# Deduce Axis Information
			current_bl_socket = self.loose_input_sockets.get('Dim')
			if (
				current_bl_socket is None
				or current_bl_socket.active_kind != ct.FlowKind.LazyValueFunc
				or current_bl_socket.mathtype != spux.MathType.Real
				or current_bl_socket.physical_type != spux.PhysicalType.NonPhysical
			):
				self.loose_input_sockets = {
					'Dim': sockets.ExprSocketDef(
						active_kind=ct.FlowKind.LazyValueFunc,
						mathtype=spux.MathType.Real,
						physical_type=spux.PhysicalType.NonPhysical,
						show_info_columns=True,
					)
				}

		# No Loose Value: Remove Input Sockets
		elif self.loose_input_sockets:
			self.loose_input_sockets = {}

	####################
	# - FlowKind.Value|LazyValueFunc
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.LazyValueFunc,
		props={'operation', 'dim_0', 'dim_1', 'slice_tuple'},
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
			and operation.are_dims_valid(info, dim_0, dim_1)
		):
			axis_0 = info.dim_names.index(dim_0) if dim_0 is not None else None
			axis_1 = info.dim_names.index(dim_1) if dim_1 is not None else None
			slice_tuple = (
				props['slice_tuple']
				if self.operation is FilterOperation.SliceIdx
				else None
			)

			return lazy_value_func.compose_within(
				operation.jax_func(axis_0, axis_1, slice_tuple),
				enclosing_func_args=operation.func_args,
				supports_jax=True,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		props={
			'dim_0',
			'dim_1',
			'operation',
			'slice_tuple',
			'set_dim_symbol',
			'set_dim_active_unit',
		},
		input_sockets={'Expr', 'Dim'},
		input_socket_kinds={
			'Expr': ct.FlowKind.Info,
			'Dim': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Params, ct.FlowKind.Info},
		},
		input_sockets_optional={'Dim': True},
	)
	def compute_info(self, props, input_sockets) -> ct.InfoFlow:
		operation = props['operation']
		info = input_sockets['Expr']
		dim_coords = input_sockets['Dim'][ct.FlowKind.LazyValueFunc]
		dim_params = input_sockets['Dim'][ct.FlowKind.Params]
		dim_info = input_sockets['Dim'][ct.FlowKind.Info]
		dim_symbol = props['set_dim_symbol']
		dim_active_unit = props['set_dim_active_unit']

		has_info = not ct.FlowSignal.check(info)
		has_dim_coords = not ct.FlowSignal.check(dim_coords)
		has_dim_params = not ct.FlowSignal.check(dim_params)
		has_dim_info = not ct.FlowSignal.check(dim_info)

		# Dimension(s)
		dim_0 = props['dim_0']
		dim_1 = props['dim_1']
		slice_tuple = props['slice_tuple']
		if has_info and operation is not None:
			# Set Dimension: Retrieve Array
			if props['operation'] is FilterOperation.SetDim:
				if (
					dim_0 is not None
					# Check Replaced Dimension
					and has_dim_coords
					and len(dim_coords.func_args) == 1
					and dim_coords.func_args[0] is spux.MathType.Integer
					and not dim_coords.func_kwargs
					and dim_coords.supports_jax
					# Check Params
					and has_dim_params
					and len(dim_params.func_args) == 1
					and not dim_params.func_kwargs
					# Check Info
					and has_dim_info
				):
					# Retrieve Dimension Coordinate Array
					## -> It must be strictly compatible.
					values = dim_coords.func_jax(int(dim_params.func_args[0]))
					if (
						len(values.shape) != 1
						or values.shape[0] != info.dim_lens[dim_0]
					):
						return ct.FlowSignal.FlowPending

					# Transform Info w/Corrected Dimension
					## -> The existing dimension will be replaced.
					if dim_active_unit is not None:
						dim_unit = spux.unit_str_to_unit(dim_active_unit)
					else:
						dim_unit = None

					new_dim_idx = ct.ArrayFlow(
						values=values,
						unit=dim_unit,
					)
					corrected_dim = [dim_0, (dim_symbol.name, new_dim_idx)]
					return operation.transform_info(
						info, dim_0, dim_1, corrected_dim=corrected_dim
					)
				return ct.FlowSignal.FlowPending
			return operation.transform_info(info, dim_0, dim_1, slice_tuple=slice_tuple)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		props={'dim_0', 'dim_1', 'operation'},
		input_sockets={'Expr', 'Value', 'Axis'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Info, ct.FlowKind.Params},
		},
		input_sockets_optional={'Value': True, 'Axis': True},
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
			and operation.are_dims_valid(info, dim_0, dim_1)
		):
			# Retrieve Pinned Value
			pinned_value = input_sockets['Value']
			has_pinned_value = not ct.FlowSignal.check(pinned_value)

			pinned_axis = input_sockets['Axis']
			has_pinned_axis = not ct.FlowSignal.check(pinned_axis)

			# Pin by-Value: Compute Nearest IDX
			## -> Presume a sorted index array to be able to use binary search.
			if props['operation'] is FilterOperation.Pin and has_pinned_value:
				nearest_idx_to_value = info.dim_idx[dim_0].nearest_idx_of(
					pinned_value, require_sorted=True
				)

				return params.compose_within(enclosing_func_args=[nearest_idx_to_value])

			# Pin by-Index
			if props['operation'] is FilterOperation.PinIdx and has_pinned_axis:
				return params.compose_within(enclosing_func_args=[pinned_axis])

			return params

		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	FilterMathNode,
]
BL_NODES = {ct.NodeType.FilterMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
