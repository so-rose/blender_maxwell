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

"""Declares `MapMathNode`."""

import typing as typ

import bpy

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import math_system, sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal
MO = math_system.MapOperation
MT = spux.MathType


class MapMathNode(base.MaxwellSimNode):
	r"""Applies a function by-structure to the data.

	The shape, type, and interpretation of the input/output data is dynamically shown.

	# Socket Sets
	The line between a "map" and a "filter" is generally a matter of taste.
	In general, "map" provides "do something to each of x" operations.

	While it is often generally assumed that `numpy` broadcasting rules are well-known, dimensional data is inherently complicated.
	Therefore, we choose an explicit, opinionated approach to "how things are mapped", prioritizing predictability over flexibility.

	## By Element
	Applies a function to each scalar number of the array.

	:::{.callout-tip title="Example"}
	Say we have a standard `(50, 3)` array with a `float32` (`f32`) datatype.
	We could interpret such an indexed structure as an **element map**:

	$$
		A:\,\,\underbrace{(\mathbb{Z}_{50}, \mathbb{Z}_3)}_{\texttt{(50,3)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}
	$$

	`By Element` simply applies a function to each output value, $\mathbb{R}$, producing a new $A$ with the same dimensions.
	Note that the datatype might be altered, ex. `\mathbb{C} \to \mathbb{R}`, as part of the function.
	:::


	## By Vector
	Applies a function to each vector, the elements of which span the **last axis**.

	This **might** produce a well-known dimensionality change, depending on what each vector maps to.

	:::{.callout-tip title="Example"}
	Let's build on the `By Element` example, by interpreting it as a list of column vectors, and taking the length of each.

	`By Vector` operates on the same data, but interpreted in a slightly deconstructed way:

	$$
		A:\,\,\underbrace{(\mathbb{Z}_{50})}_{\texttt{(50,)}} \to (\underbrace{(\mathbb{Z}_3)}_{\texttt{(3,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}})
	$$

	`By Vector` applies a function to each $\underbrace{(\mathbb{Z}_3)}_{\texttt{(3,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}$.
	Applying a standard 2-norm

	$$
		||\cdot||_2:\,\,\,\,(\underbrace{(\mathbb{Z}_3)}_{\texttt{(3,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}) \to \underbrace{(\mathbb{R})}_{\texttt{f32}}
	$$

	to our $A$ results in a new, reduced-dimension array:

	$$
		A_{||\cdot||_2}:\,\,\underbrace{(\mathbb{Z}_{50})}_{\texttt{(50,)}} \to \underbrace{(\mathbb{R})}_{\texttt{f32}}
	$$
	:::


	## By Matrix
	Applies a function to each matrix, the elements of which span the **last two axes**.

	This **might** produce a well-known dimensionality change, depending on what each matrix maps to.

	:::{.callout-tip title="Just Like Vectors"}
	At this point, we reach 3D, and mental models become more difficult.

	When dealing with high-dimensional arrays, it is suggested to draw out the math, ex. with the explicit notation introduced earlier.
	:::

	## Expr
	Applies a user-sourced symbolic expression to a single symbol, with the symbol either representing (selectably) a single element, vector, or matrix.
	The name and type of the available symbol is clearly shown, and most valid `sympy` expressions that you would expect to work, should work.

	Use of expressions generally imposes no performance penalty: Just like the baked-in operations, it is compiled to a high-performance `jax` function.
	Thus, it participates in the `FK.Func` composition chain.


	Attributes:
		operation: Operation to apply to the input.
	"""

	node_type = ct.NodeType.MapMath
	bl_label = 'Map Math'

	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=FK.Func),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=FK.Func),
	}

	####################
	# - Properties: Incoming InfoFlow
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
	# - Property: Operation
	####################
	operation: MO = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_info'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		"""Retrieve valid operations based on the input `InfoFlow`."""
		if self.expr_info is not None:
			return MO.bl_enum_elements(self.expr_info)
		return []

	####################
	# - UI
	####################
	def draw_label(self):
		"""Show the current operation (if any) in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		if self.operation is not None:
			return 'Map: ' + MO.to_name(self.operation)

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Draw the user interfaces of the node's properties inside of the node itself.

		Parameters:
			layout: UI target for drawing.
		"""
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - FlowKind.Value
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Value,
		# Loaded
		props={'operation'},
		inscks_kinds={'Expr': FK.Value},
	)
	def compute_value(self, props, input_sockets) -> ct.ValueFlow | FS:
		"""Mapping operation on symbolic input expression."""
		expr = input_sockets['Expr']

		operation = props['operation']
		if operation is not None:
			return operation.sp_func(expr)
		return FS.FlowPending

	####################
	# - FlowKind.Func
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
		"""Mapping operation on lazy-defined input expression."""
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
		props={'operation'},
		inscks_kinds={
			'Expr': FK.Info,
		},
	)
	def compute_info(self, props, input_sockets) -> ct.InfoFlow:
		"""Transform the info chracterization of the input."""
		info = input_sockets['Expr']

		operation = props['operation']
		if operation is not None:
			return operation.transform_info(info)
		return FS.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=FK.Params,
		# Loaded
		props={'operation'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': FK.Params},
	)
	def compute_params(self, props, input_sockets) -> ct.ParamsFlow | FS:
		"""Transform the parameters of the input."""
		params = input_sockets['Expr']

		operation = props['operation']
		if operation is not None:
			return operation.transform_params(params)
		return FS.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	MapMathNode,
]
BL_NODES = {ct.NodeType.MapMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
