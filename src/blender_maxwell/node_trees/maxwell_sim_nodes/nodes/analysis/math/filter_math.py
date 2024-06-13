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
import sympy as sp

from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import math_system, sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
FO = math_system.FilterOperation
MT = spux.MathType


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
		'Expr': sockets.ExprSocketDef(active_kind=FK.Func, show_func_ui=False),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=FK.Func),
	}

	####################
	# - Properties: Expr InfoFlow
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Expr': FK.Info},
		# Loaded
		inscks_kinds={'Expr': FK.Info},
		input_sockets_optional={'Expr'},
		# Flow
		## -> See docs in TransformMathNode
		stop_propagation=True,
	)
	def on_input_expr_changed(self, input_sockets) -> None:  # noqa: D102
		info = input_sockets['Expr']
		has_info = not FS.check(info)
		info_pending = FS.check_single(info, FS.FlowPending)

		if has_info and not info_pending:
			self.expr_info = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def expr_info(self) -> ct.InfoFlow | None:
		"""Retrieve the input expression's `InfoFlow`."""
		info = self._compute_input('Expr', kind=FK.Info)
		has_info = not FS.check(info)
		if has_info:
			return info
		return None

	####################
	# - Properties: Operation
	####################
	operation: FO = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_info'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		"""Determine all valid operations from the input expression."""
		if self.expr_info is not None:
			return FO.bl_enum_elements(self.expr_info)
		return []

	####################
	# - Properties: Dimension Selection
	####################
	active_dim_0: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_dims(),
		cb_depends_on={'operation', 'expr_info'},
	)
	active_dim_1: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_dims(),
		cb_depends_on={'operation', 'expr_info'},
	)

	def search_dims(self) -> list[ct.BLEnumElement]:
		"""Determine all valid dimensions from the input expression."""
		if self.expr_info is not None and self.operation is not None:
			return [
				(dim.name, dim.name_pretty, dim.name, '', i)
				for i, dim in enumerate(self.operation.valid_dims(self.expr_info))
			]
		return []

	@bl_cache.cached_bl_property(depends_on={'active_dim_0'})
	def dim_0(self) -> sim_symbols.SimSymbol | None:
		"""The first currently active dimension, if any is selected; otherwise `None`."""
		if self.expr_info is not None and self.active_dim_0 is not None:
			return self.expr_info.dim_by_name(self.active_dim_0)
		return None

	@bl_cache.cached_bl_property(depends_on={'active_dim_1'})
	def dim_1(self) -> sim_symbols.SimSymbol | None:
		"""The second currently active dimension, if any is selected; otherwise `None`."""
		if self.expr_info is not None and self.active_dim_1 is not None:
			return self.expr_info.dim_by_name(self.active_dim_1)
		return None

	@bl_cache.cached_bl_property(depends_on={'dim_0'})
	def axis_0(self) -> sim_symbols.SimSymbol | None:
		"""The first currently active axis, derived from `self.dim_0`."""
		if self.expr_info is not None and self.dim_0 is not None:
			return self.expr_info.dim_axis(self.dim_0)
		return None

	@bl_cache.cached_bl_property(depends_on={'dim_1'})
	def axis_1(self) -> sim_symbols.SimSymbol | None:
		"""The second currently active dimension, if any is selected; otherwise `None`."""
		if self.expr_info is not None and self.active_dim_1 is not None:
			return self.expr_info.dim_axis(self.dim_1)
		return None

	####################
	# - Properties: Slice
	####################
	slice_tuple: tuple[int, int, int] = bl_cache.BLField([0, 1, 1])

	####################
	# - UI
	####################
	def draw_label(self):  # noqa: PLR0911
		"""Show the active filter operation in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		match self.operation:
			# Slice
			case FO.SliceIdx:
				slice_str = ':'.join([str(v) for v in self.slice_tuple])
				return f'Filter: {self.active_dim_0}[{slice_str}]'

			# Pin
			case FO.PinLen1:
				return f'Filter: Pin {self.active_dim_0}[0]'
			case FO.Pin:
				return f'Filter: Pin {self.active_dim_0}[...]'
			case FO.PinIdx:
				pin_idx_axis = self._compute_input('Index', kind=FK.Value)
				has_pin_idx_axis = not FS.check(pin_idx_axis)
				if has_pin_idx_axis:
					return f'Filter: Pin {self.active_dim_0}[{pin_idx_axis}]'
				return self.bl_label

			# Reinterpret
			case FO.Swap:
				return f'Filter: Swap [{self.active_dim_0}]|[{self.active_dim_1}]'

			case _:
				return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw the user interfaces of the node's properties inside of the node itself.

		Parameters:
			layout: UI target for drawing.
		"""
		layout.prop(self, self.blfields['operation'], text='')

		if self.operation is not None:
			match self.operation.num_dim_inputs:
				case 1:
					layout.prop(self, self.blfields['active_dim_0'], text='')
				case 2:
					row = layout.row(align=True)
					row.prop(self, self.blfields['active_dim_0'], text='')
					row.prop(self, self.blfields['active_dim_1'], text='')

			if self.operation is FO.SliceIdx:
				layout.prop(self, self.blfields['slice_tuple'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Expr': FK.Info},
		prop_name={'operation', 'dim_0'},
		# Loaded
		props={'operation', 'dim_0'},
		inscks_kinds={'Expr': FK.Info},
		input_sockets_optional={'Expr'},
	)
	def on_pin_factors_changed(self, props, input_sockets) -> None:
		"""Synchronize loose input sockets to match the dimension-pinning method declared in `self.operation`."""
		info = input_sockets['Expr']
		has_info = not FS.check(info)

		dim_0 = props['dim_0']

		operation = props['operation']
		match operation:
			case FO.Pin if (
				has_info
				and dim_0 is not None
				and (info.has_idx_cont(dim_0) or info.has_idx_discrete(dim_0))
			):
				self.loose_input_sockets = {
					'Value': sockets.ExprSocketDef(
						active_kind=FK.Value,
						**dim_0.expr_info,
					),
				}

			case FO.PinIdx if (
				has_info
				and dim_0 is not None
				and (info.has_idx_labels(dim_0) or info.has_idx_discrete(dim_0))
			):
				self.loose_input_sockets = {
					'Index': sockets.ExprSocketDef(
						active_kind=FK.Value,
						output_name=dim_0.expr_info['output_name'],
						mathtype=MT.Integer,
						abs_min=0,
						abs_max=len(info.dims[dim_0]) - 1,
					),
				}

			case _ if self.loose_input_sockets:
				self.loose_input_sockets = {}

	####################
	# - FlowKind.Func
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Func,
		# Loaded
		props={'operation', 'axis_0', 'axis_1', 'slice_tuple'},
		inscks_kinds={'Expr': FK.Func},
	)
	def compute_func(self, props, input_sockets) -> None:
		"""Filter operation on lazy-defined input expression."""
		lazy_func = input_sockets['Expr']

		axis_0 = props['axis_0']
		axis_1 = props['axis_1']
		slice_tuple = props['slice_tuple']

		operation = props['operation']
		if operation is not None:
			new_func = operation.transform_func(
				lazy_func, axis_0, axis_1=axis_1, slice_tuple=slice_tuple
			)
			if new_func is not None:
				return new_func

		return FS.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Info,
		# Loaded
		props={
			'dim_0',
			'dim_1',
			'operation',
			'slice_tuple',
		},
		inscks_kinds={
			'Expr': FK.Info,
			'Value': {FK.Func, FK.Params},
			'Index': {FK.Func, FK.Params},
		},
		input_sockets_optional={'Index', 'Value'},
	)
	def compute_info(self, props, input_sockets) -> ct.InfoFlow:
		"""Transform `InfoFlow` based on the current filtering operation."""
		info = input_sockets['Expr']

		dim_0 = props['dim_0']
		dim_1 = props['dim_1']

		operation = props['operation']
		match operation:
			# Slice
			case FO.Slice | FO.SliceIdx if dim_0 is not None:
				slice_tuple = props['slice_tuple']
				return operation.transform_info(info, dim_0, slice_tuple=slice_tuple)

			# Pin
			case FO.PinLen1 if dim_0 is not None:
				return operation.transform_info(info, dim_0)

			case FO.Pin if dim_0 is not None:
				pinned_value = events.realize_known(
					input_sockets['Value'], conformed=True
				)
				if pinned_value is not None:
					nearest_idx_to_value = info.dims[dim_0].nearest_idx_of(
						pinned_value, require_sorted=True
					)
					return operation.transform_info(
						info, dim_0, pin_idx=nearest_idx_to_value
					)

			case FO.PinIdx if dim_0 is not None:
				pinned_idx = int(events.realize_known(input_sockets['Index']))
				if pinned_idx is not None:
					return operation.transform_info(info, dim_0, pin_idx=pinned_idx)

			# Swizzle
			case FO.Swap if dim_0 is not None and dim_1 is not None:
				return operation.transform_info(info, dim_0, dim_1)

		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Params,
		# Loaded
		props={'dim_0', 'dim_1', 'operation'},
		inscks_kinds={
			'Value': {FK.Func, FK.Params},
			'Index': {FK.Func, FK.Params},
			'Expr': {FK.Info, FK.Params},
		},
		input_sockets_optional={'Value', 'Index'},
	)
	def compute_params(self, props, input_sockets) -> ct.ParamsFlow:
		"""Compute tracked function argument parameters of input parameters."""
		info = input_sockets['Expr'][FK.Info]
		params = input_sockets['Expr'][FK.Params]

		dim_0 = props['dim_0']

		operation = props['operation']
		match operation:
			# *
			case FO.Slice | FO.SliceIdx | FO.PinLen1 | FO.Swap:
				return params

			# Pin
			case FO.Pin:
				pinned_value = events.realize_known(
					input_sockets['Value'], conformed=True
				)
				if pinned_value is not None:
					nearest_idx_to_value = info.dims[dim_0].nearest_idx_of(
						pinned_value, require_sorted=True
					)
					return params.compose_within(
						enclosing_func_args=(sp.Integer(nearest_idx_to_value),),
					)

			case FO.PinIdx:
				pinned_idx = events.realize_known(input_sockets['Index'])
				if pinned_idx is not None:
					return params.compose_within(
						enclosing_func_args=(sp.Integer(pinned_idx),),
					)

		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	FilterMathNode,
]
BL_NODES = {ct.NodeType.FilterMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
