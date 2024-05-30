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

import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu

from blender_maxwell.utils import bl_cache, sci_constants, sim_symbols
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events


class ScientificConstantNode(base.MaxwellSimNode):
	"""A well-known constant usable as itself, or as a symbol."""

	node_type = ct.NodeType.ScientificConstant
	bl_label = 'Scientific Constant'

	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Func),
	}

	####################
	# - Properties
	####################
	use_symbol: bool = bl_cache.BLField(False)

	sci_constant_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.LowerU
	)
	sci_constant_str: str = bl_cache.BLField(
		'',
		str_cb=lambda self, _, edit_text: self.search_sci_constants(edit_text),
	)

	def search_sci_constants(
		self,
		edit_text: str,
	):
		return [
			name
			for name in sci_constants.SCI_CONSTANTS
			if edit_text.lower() in name.lower()
		]

	@bl_cache.cached_bl_property(depends_on={'sci_constant_str'})
	def sci_constant(self) -> spux.SympyExpr | None:
		"""Retrieve the expression for the scientific constant."""
		return sci_constants.SCI_CONSTANTS.get(self.sci_constant_str)

	@bl_cache.cached_bl_property(depends_on={'sci_constant_str'})
	def sci_constant_info(self) -> spux.SympyExpr | None:
		"""Retrieve the information for the selected scientific constant."""
		return sci_constants.SCI_CONSTANTS_INFO.get(self.sci_constant_str)

	@bl_cache.cached_bl_property(
		depends_on={'sci_constant', 'sci_constant_info', 'sci_constant_name'}
	)
	def sci_constant_sym(self) -> spux.SympyExpr | None:
		"""Retrieve a symbol for the scientific constant."""
		if self.sci_constant is not None and self.sci_constant_info is not None:
			unit = self.sci_constant_info['units']
			return sim_symbols.SimSymbol.from_expr(
				self.sci_constant_name,
				self.sci_constant,
				unit,
				is_constant=True,
			)

		return None

	####################
	# - UI
	####################
	def draw_label(self):
		if self.sci_constant_str:
			return f'Const: {self.sci_constant_str}'
		return self.bl_label

	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		col.prop(self, self.blfields['sci_constant_str'], text='')

		row = col.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='Assign Symbol')
		col.prop(self, self.blfields['sci_constant_name'], text='')
		col.prop(self, self.blfields['use_symbol'], text='Use Symbol', toggle=True)

	def draw_info(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		box = col.box()
		split = box.split(factor=0.25, align=True)

		# Left: Units
		_col = split.column(align=True)
		row = _col.row(align=True)
		# row.alignment = 'CENTER'
		row.label(text='Src')

		if self.sci_constant_info:
			row = _col.row(align=True)
			# row.alignment = 'CENTER'
			row.label(text='Unit')

			row = _col.row(align=True)
			# row.alignment = 'CENTER'
			row.label(text='Err')

		# Right: Values
		_col = split.column(align=True)
		row = _col.row(align=True)
		# row.alignment = 'CENTER'
		row.label(text='CODATA2018')

		if self.sci_constant_info:
			row = _col.row(align=True)
			# row.alignment = 'CENTER'
			row.label(text=f'{spux.sp_to_str(self.sci_constant_info["units"].n(4))}')

			row = _col.row(align=True)
			# row.alignment = 'CENTER'
			row.label(text=f'{self.sci_constant_info["uncertainty"]}')

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'Expr',
		props={'use_symbol', 'sci_constant', 'sci_constant_sym'},
	)
	def compute_value(self, props) -> typ.Any:
		sci_constant = props['sci_constant']
		sci_constant_sym = props['sci_constant_sym']

		if props['use_symbol'] and sci_constant_sym is not None:
			return sci_constant_sym.sp_symbol

		if sci_constant is not None:
			return sci_constant

		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Func,
		props={'sci_constant', 'sci_constant_sym'},
	)
	def compute_lazy_func(self, props) -> typ.Any:
		"""Simple `FuncFlow` that computes the symbol value, with output units tracked correctly."""
		sci_constant = props['sci_constant']
		sci_constant_sym = props['sci_constant_sym']

		if sci_constant is not None:
			return ct.FuncFlow(
				func=sp.lambdify(
					[sci_constant_sym.sp_symbol], sci_constant_sym.sp_symbol, 'jax'
				),
				func_args=[sci_constant_sym],
				func_output=sci_constant_sym,
				supports_jax=True,
			)
		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		props={'sci_constant_sym'},
	)
	def compute_info(self, props: dict) -> typ.Any:
		"""Simple `FuncFlow` that computes the symbol value, with output units tracked correctly."""
		sci_constant_sym = props['sci_constant_sym']

		if sci_constant_sym is not None:
			return ct.InfoFlow(output=sci_constant_sym)
		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		props={'sci_constant', 'sci_constant_sym'},
	)
	def compute_params(self, props: dict) -> typ.Any:
		sci_constant = props['sci_constant']
		sci_constant_sym = props['sci_constant_sym']

		if sci_constant is not None and sci_constant_sym is not None:
			return ct.ParamsFlow(
				arg_targets=[sci_constant_sym],
				func_args=[sci_constant_sym.sp_symbol],
				symbols={sci_constant_sym},
			).realize_partial(
				{
					sci_constant_sym: sci_constant,
				}
			)
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	ScientificConstantNode,
]
BL_NODES = {
	ct.NodeType.ScientificConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)
}
