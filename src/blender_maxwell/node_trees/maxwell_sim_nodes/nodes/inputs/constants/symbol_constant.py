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
from fractions import Fraction

import bpy
import sympy as sp

from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class SymbolConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SymbolConstant
	bl_label = 'Symbol'

	input_sockets: typ.ClassVar = {}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.Func,
			show_info_columns=True,
		),
	}

	####################
	# - Socket Interface
	####################
	sym_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.Constant
	)

	size: spux.NumberSize1D = bl_cache.BLField(spux.NumberSize1D.Scalar)
	## Use of NumberSize1D implicitly guarantees UI-realizability later.

	mathtype: spux.MathType = bl_cache.BLField(spux.MathType.Real)
	physical_type: spux.PhysicalType = bl_cache.BLField(spux.PhysicalType.NonPhysical)

	####################
	# - Properties: Unit
	####################
	active_unit: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_valid_units(),
		cb_depends_on={'physical_type'},
	)

	def search_valid_units(self) -> list[ct.BLEnumElement]:
		"""Compute Blender enum elements of valid units for the current `physical_type`."""
		if self.physical_type is not spux.PhysicalType.NonPhysical:
			return [
				(sp.sstr(unit), spux.sp_to_str(unit), sp.sstr(unit), '', i)
				for i, unit in enumerate(self.physical_type.valid_units)
			]
		return []

	@bl_cache.cached_bl_property(depends_on={'active_unit'})
	def unit(self) -> spux.Unit | None:
		"""Gets the current active unit.

		Returns:
			The current active `sympy` unit.

			If the socket expression is unitless, this returns `None`.
		"""
		if self.active_unit is not None:
			return spux.unit_str_to_unit(self.active_unit)

		return None

	@property
	def unit_factor(self) -> spux.Unit | None:
		"""Like `self.unit`, except `1` instead of `None` when unitless."""
		return sp.Integer(1) if self.unit is None else self.unit

	####################
	# - Domain
	####################
	interval_finite_z: tuple[int, int] = bl_cache.BLField((0, 1))
	interval_finite_q: tuple[tuple[int, int], tuple[int, int]] = bl_cache.BLField(
		((0, 1), (1, 1))
	)
	interval_finite_re: tuple[float, float] = bl_cache.BLField((0.0, 1.0))
	interval_inf: tuple[bool, bool] = bl_cache.BLField((True, True))
	interval_closed: tuple[bool, bool] = bl_cache.BLField((True, True))

	interval_finite_im: tuple[float, float] = bl_cache.BLField((0.0, 1.0))
	interval_inf_im: tuple[bool, bool] = bl_cache.BLField((True, True))
	interval_closed_im: tuple[bool, bool] = bl_cache.BLField((True, True))

	preview_value_z: int = bl_cache.BLField(0)
	preview_value_q: tuple[int, int] = bl_cache.BLField((0, 1))
	preview_value_re: float = bl_cache.BLField(0.0)
	preview_value_im: float = bl_cache.BLField(0.0)

	@bl_cache.cached_bl_property(
		depends_on={
			'mathtype',
			'interval_finite_z',
			'interval_finite_q',
			'interval_finite_re',
			'interval_finite_im',
		}
	)
	def interval_finite(
		self,
	) -> (
		tuple[int | Fraction | float, int | Fraction | float]
		| tuple[tuple[float, float], tuple[float, float]]
	):
		"""Return the appropriate finite interval from the UI, as guided by `self.mathtype`."""
		MT = spux.MathType
		match self.mathtype:
			case MT.Integer:
				return self.interval_finite_z
			case MT.Rational:
				return [Fraction(*q) for q in self.interval_finite_q]
			case MT.Real:
				return self.interval_finite_re
			case MT.Complex:
				return (self.interval_finite_re, self.interval_finite_im)

	@bl_cache.cached_bl_property(
		depends_on={
			'mathtype',
			'preview_value_z',
			'preview_value_q',
			'preview_value_re',
			'preview_value_im',
		}
	)
	def preview_value(
		self,
	) -> int | Fraction | float | complex:
		"""Return the appropriate finite interval from the UI, as guided by `self.mathtype`."""
		MT = spux.MathType
		match self.mathtype:
			case MT.Integer:
				return self.preview_value_z
			case MT.Rational:
				return Fraction(*self.preview_value_q)
			case MT.Real:
				return self.preview_value_re
			case MT.Complex:
				return complex(self.preview_value_re, self.preview_value_im)

	@bl_cache.cached_bl_property(
		depends_on={
			'mathtype',
			'interval_finite',
			'interval_inf',
			'interval_inf_im',
			'interval_closed',
			'interval_closed_im',
		}
	)
	def domain(
		self,
	) -> sp.Interval | sp.sets.fancysets.CartesianComplexRegion:
		"""Deduce the domain specified in the UI."""
		MT = spux.MathType
		match self.mathtype:
			case MT.Integer | MT.Real | MT.Rational:
				return sim_symbols.mk_interval(
					self.interval_finite,
					self.interval_inf,
					self.interval_closed,
				)

			case MT.Complex:
				region = self.interval_finite
				domain_re = sim_symbols.mk_interval(
					region[0],
					self.interval_inf,
					self.interval_closed,
				)
				domain_im = sim_symbols.mk_interval(
					region[1],
					self.interval_inf_im,
					self.interval_closed_im,
				)
				return sp.ComplexRegion(domain_re, domain_im, polar=False)

	####################
	# - Computed Properties
	####################
	@bl_cache.cached_bl_property(
		depends_on={
			'sym_name',
			'mathtype',
			'physical_type',
			'unit',
			'size',
			'domain',
			'preview_value',
		}
	)
	def symbol(self) -> sim_symbols.SimSymbol:
		"""Generate the `SimSymbol` matching the user-specification."""
		return sim_symbols.SimSymbol(
			sym_name=self.sym_name,
			mathtype=self.mathtype,
			physical_type=self.physical_type,
			unit=self.unit,
			rows=self.size.rows,
			cols=self.size.cols,
			domain=self.domain,
			preview_value=self.preview_value,
		)

	####################
	# - UI
	####################
	def draw_label(self):
		return self.symbol.def_label

	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		col.prop(self, self.blfields['sym_name'], text='')

		row = col.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='Assumptions')

		row = col.row(align=True)
		row.prop(self, self.blfields['mathtype'], text='')

		row = col.row(align=True)
		row.prop(self, self.blfields['size'], text='')
		row.prop(self, self.blfields['active_unit'], text='')

		col.prop(self, self.blfields['physical_type'], text='')

		# Domain - Infinite
		row = col.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='Domain - Is Infinite')

		row = col.row(align=True)
		if self.mathtype is spux.MathType.Complex:
			row.prop(self, self.blfields['interval_inf'], text='ℝ')
			row.prop(self, self.blfields['interval_inf_im'], text='𝕀')
		else:
			row.prop(self, self.blfields['interval_inf'], text='')

		if any(not b for b in self.interval_inf):
			# Domain - Closure
			row = col.row(align=True)
			row.alignment = 'CENTER'
			row.label(text='Domain - Closure')

			row = col.row(align=True)
			if self.mathtype is spux.MathType.Complex:
				row.prop(self, self.blfields['interval_closed'], text='ℝ')
				row.prop(self, self.blfields['interval_closed_im'], text='𝕀')
			else:
				row.prop(self, self.blfields['interval_closed'], text='')

			# Domain - Finite
			row = col.row(align=True)
			row.alignment = 'CENTER'
			row.label(text='Domain - Interval')

			row = col.row(align=True)
			match self.mathtype:
				case spux.MathType.Integer:
					row.prop(self, self.blfields['interval_finite_z'], text='')

				case spux.MathType.Rational:
					row.prop(self, self.blfields['interval_finite_q'], text='')

				case spux.MathType.Real:
					row.prop(self, self.blfields['interval_finite_re'], text='')

				case spux.MathType.Complex:
					row.prop(self, self.blfields['interval_finite_re'], text='ℝ')
					row.prop(self, self.blfields['interval_finite_im'], text='𝕀')

		# Domain - Closure
		row = col.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='Preview Value')
		match self.mathtype:
			case spux.MathType.Integer:
				row.prop(self, self.blfields['preview_value_z'], text='')

			case spux.MathType.Rational:
				row.prop(self, self.blfields['preview_value_q'], text='')

			case spux.MathType.Real:
				row.prop(self, self.blfields['preview_value_re'], text='')

			case spux.MathType.Complex:
				row.prop(self, self.blfields['preview_value_re'], text='ℝ')
				row.prop(self, self.blfields['preview_value_im'], text='𝕀')

	####################
	# - FlowKinds
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Value,
		# Loaded
		props={'symbol'},
	)
	def compute_value(self, props) -> typ.Any:
		return props['symbol'].sp_symbol

	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'symbol'},
	)
	def compute_lazy_func(self, props) -> typ.Any:
		sym = props['symbol']
		return ct.FuncFlow(
			func=sp.lambdify(sym.sp_symbol_matsym, sym.sp_symbol_matsym, 'jax'),
			func_args=[sym],
			func_output=sym,
			supports_jax=True,
		)

	####################
	# - FlowKinds: Auxiliary
	####################
	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Info,
		# Loaded
		props={'symbol'},
	)
	def compute_info(self, props) -> typ.Any:
		return ct.InfoFlow(
			dims={props['symbol']: None},
			output=props['symbol'],
		)

	@events.computes_output_socket(
		# Trigger
		'Expr',
		kind=ct.FlowKind.Params,
		# Loaded
		props={'symbol'},
	)
	def compute_params(self, props) -> typ.Any:
		sym = props['symbol']
		return ct.ParamsFlow(
			arg_targets=[sym],
			func_args=[sym.sp_symbol],
			symbols={sym},
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	SymbolConstantNode,
]
BL_NODES = {ct.NodeType.SymbolConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)}
