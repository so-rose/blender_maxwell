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

"""Declares `ReduceMathNode`."""

import enum
import typing as typ

import bpy
import jax
import jax.numpy as jnp
import numpy as np

from blender_maxwell.utils import bl_cache, logger, sim_symbols

from .... import contracts as ct
from .... import math_system, sockets
from ... import base, events

log = logger.get(__name__)
FK = ct.FlowKind
FS = ct.FlowSignal
RO = math_system.ReduceOperation


class ReduceMathNode(base.MaxwellSimNode):
	r"""Applies a function to the array as a whole, with arbitrary results.

	The shape, type, and interpretation of the input/output data is dynamically shown.

	Attributes:
		operation: Operation to apply to the input.
	"""

	node_type = ct.NodeType.ReduceMath
	bl_label = 'Reduce Math'

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
		## -> Expr wants to emit DataChanged, which is usually fine.
		## -> However, this node sets `expr_info`, which causes DC to emit.
		## -> One action should emit one DataChanged pipe.
		## -> Therefore, defer responsibility for DataChanged to self.expr_info.
		# stop_propagation=True,
	)
	def on_input_exprs_changed(self, input_sockets) -> None:  # noqa: D102
		has_info = not FS.check(input_sockets['Expr'])
		info_pending = FS.check_single(input_sockets['Expr'], FS.FlowPending)

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
	operation: RO = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_info'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		"""Retrieve valid operations based on the input `InfoFlow`."""
		if self.expr_info is not None:
			return RO.bl_enum_elements(self.expr_info)
		return []

	####################
	# - Properties: Dimension Selection
	####################
	active_dim: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_dims(),
		cb_depends_on={'operation', 'expr_info'},
	)

	def search_dims(self) -> list[ct.BLEnumElement]:
		"""Search valid dimensions for reduction."""
		if self.expr_info is not None and self.operation is not None:
			return [
				(dim.name, dim.name_pretty, dim.name, '', i)
				for i, dim in enumerate(self.operation.valid_dims(self.expr_info))
			]
		return []

	@bl_cache.cached_bl_property(depends_on={'expr_info', 'active_dim'})
	def dim(self) -> sim_symbols.SimSymbol | None:
		"""Deduce the valid dimension."""
		if self.expr_info is not None and self.active_dim is not None:
			return self.expr_info.dim_by_name(self.active_dim, optional=True)
		return None

	@bl_cache.cached_bl_property(depends_on={'dim_0'})
	def axis(self) -> int | None:
		"""The first currently active axis, derived from `self.dim_0`."""
		if self.expr_info is not None and self.dim is not None:
			return self.expr_info.dim_axis(self.dim)
		return None

	####################
	# - UI
	####################
	def draw_label(self):
		"""Show the active reduce operation in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		if self.operation is not None:
			if self.dim is not None:
				return self.operation.name.replace('[a]', f'[{self.dim.name_pretty}]')
			return self.operation.name
		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw the user interfaces of the node's properties inside of the node itself.

		Parameters:
			layout: UI target for drawing.
		"""
		layout.prop(self, self.blfields['operation'], text='')
		layout.prop(self, self.blfields['active_dim'], text='')

	####################
	# - Compute: Array
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Array,
		# Loaded
		outscks_kinds={
			'Expr': {FK.Func, FK.Params},
		},
	)
	def compute_array(self, output_sockets) -> ct.ArrayFlow | FS:
		"""Realize an `ArrayFlow` containing the array."""
		array = events.realize_known(output_sockets['Expr'])
		if array is not None:
			return ct.ArrayFlow(
				jax_bytes=(
					array
					if isinstance(array, np.ndarray | jax.Array)
					else jnp.array(array)
				),
				unit=output_sockets['Expr'][FK.Func].func_output.unit,
				is_sorted=True,
			)
		return FS.FlowPending

	####################
	# - Compute: Func
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Func,
		# Loaded
		props={'operation'},
		inscks_kinds={
			'Expr': FK.Func,
		},
	)
	def compute_func(self, props, input_sockets) -> ct.FuncFlow | FS:
		"""Transform the input `FuncFlow` depending on the reduce operation."""
		func = input_sockets['Expr']

		operation = props['operation']
		if operation is not None:
			return operation.transform_func(func)
		return FS.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Info,
		# Loaded
		props={'operation', 'dim', 'expr_info'},
	)
	def compute_info(self, props) -> ct.InfoFlow | FS:
		"""Transform the input `InfoFlow` depending on the reduce operation."""
		info = props['expr_info']
		dim = props['dim']

		operation = props['operation']
		if operation is not None:
			return operation.transform_info(info, dim)
		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Params,
		# Loaded
		props={'operation', 'axis'},
		inscks_kinds={'Expr': FK.Params},
	)
	def compute_params(self, props, input_sockets) -> ct.ParamsFlow | FS:
		"""Transform the input `InfoFlow` depending on the reduce operation."""
		params = input_sockets['Expr']
		axis = props['axis']

		operation = props['operation']
		if operation is not None and axis is not None:
			return operation.transform_params(params, axis)
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	ReduceMathNode,
]
BL_NODES = {ct.NodeType.ReduceMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
