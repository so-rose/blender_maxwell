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

"""Declares `TransformMathNode`."""

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


class TransformMathNode(base.MaxwellSimNode):
	r"""Applies a function to the array as a whole, with arbitrary results.

	The shape, type, and interpretation of the input/output data is dynamically shown.

	# Socket Sets
	## Interpret
	Reinterprets the `InfoFlow` of an array, **without changing it**.

	Attributes:
		operation: Operation to apply to the input.
	"""

	node_type = ct.NodeType.TransformMath
	bl_label = 'Transform Math'

	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Func),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Func),
	}

	####################
	# - Properties: Expr InfoFlow
	####################
	@events.on_value_changed(
		# Trigger
		socket_name={'Expr'},
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
		input_sockets_optional={'Expr': True},
		# Flow
		## -> Expr wants to emit DataChanged, which is usually fine.
		## -> However, this node sets `expr_info`, which causes DC to emit.
		## -> One action should emit one DataChanged pipe.
		## -> Therefore, defer responsibility for DataChanged to self.expr_info.
		stop_propagation=True,
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
	operation: math_system.TransformOperation = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_info'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.expr_info is not None:
			return [
				operation.bl_enum_element(i)
				for i, operation in enumerate(
					math_system.TransformOperation.by_info(self.expr_info)
				)
			]
		return []

	####################
	# - Properties: Dimension Selection
	####################
	active_dim: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_dims(),
		cb_depends_on={'operation', 'expr_info'},
	)

	def search_dims(self) -> list[ct.BLEnumElement]:
		if self.expr_info is not None and self.operation is not None:
			return [
				(dim.name, dim.name_pretty, dim.name, '', i)
				for i, dim in enumerate(self.operation.valid_dims(self.expr_info))
			]
		return []

	@bl_cache.cached_bl_property(depends_on={'expr_info', 'active_dim'})
	def dim(self) -> sim_symbols.SimSymbol | None:
		if self.expr_info is not None and self.active_dim is not None:
			return self.expr_info.dim_by_name(self.active_dim, optional=True)
		return None

	####################
	# - Properties: New Dimension Properties
	####################
	new_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.Expr
	)
	new_physical_type: spux.PhysicalType = bl_cache.BLField(
		spux.PhysicalType.NonPhysical
	)
	active_new_unit: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_units(),
		cb_depends_on={'dim', 'new_physical_type', 'operation'},
	)

	def search_units(self) -> list[ct.BLEnumElement]:
		TO = math_system.TransformOperation
		match self.operation:
			# Covariant Transform
			case TO.ConvertIdxUnit if self.dim is not None:
				physical_type = spux.PhysicalType.from_unit(
					self.dim.unit, optional=True
				)
				if physical_type is not None:
					valid_units = physical_type.valid_units
				else:
					valid_units = []

			case TO.FreqToVacWL if self.dim is not None:
				valid_units = spux.PhysicalType.Length.valid_units

			case TO.VacWLToFreq if self.dim is not None:
				valid_units = spux.PhysicalType.Freq.valid_units

			case TO.SetIdxUnit if (
				self.dim is not None
				and self.new_physical_type is not spux.PhysicalType.NonPhysical
			):
				valid_units = self.new_physical_type.valid_units

			case TO.FirstColToFirstIdx if (
				self.new_physical_type is not spux.PhysicalType.NonPhysical
			):
				valid_units = self.new_physical_type.valid_units

			case _:
				valid_units = []

		return [
			(
				sp.sstr(unit),
				spux.sp_to_str(unit),
				sp.sstr(unit),
				'',
				i,
			)
			for i, unit in enumerate(valid_units)
		]

	@bl_cache.cached_bl_property(depends_on={'active_new_unit'})
	def new_unit(self) -> spux.Unit:
		if self.active_new_unit is not None:
			return spux.unit_str_to_unit(self.active_new_unit)

		return None

	####################
	# - UI
	####################
	@bl_cache.cached_bl_property(depends_on={'new_unit'})
	def new_unit_str(self) -> str:
		if self.new_unit is None:
			return ''
		return spux.sp_to_str(self.new_unit)

	def draw_label(self):
		TO = math_system.TransformOperation
		match self.operation:
			case TO.FreqToVacWL if self.dim is not None:
				return f'T: {self.dim.name_pretty} | 𝑓 → {self.new_unit_str}'

			case TO.VacWLToFreq if self.dim is not None:
				return f'T: {self.dim.name_pretty} | λᵥ → {self.new_unit_str}'

			case TO.ConvertIdxUnit if self.dim is not None:
				return f'T: {self.dim.name_pretty} → {self.new_unit_str}'

			case TO.SetIdxUnit if self.dim is not None:
				return f'T: {self.dim.name_pretty} → {self.new_name.name_pretty} | {self.new_unit_str}'

			case (
				TO.IntDimToComplex
				| TO.DimToVec
				| TO.DimsToMat
			) if self.expr_info is not None and self.expr_info.dims:
				return f'T: {self.expr_info.last_dim.name_unit_label} {self.operation.name}'

			case TO.FT1D if self.dim is not None:
				return f'T: FT[{self.dim.name_unit_label}]'

			case TO.InvFT1D if self.dim is not None:
				return f'T: iFT[{self.dim.name_unit_label}]'

			case _:
				if self.operation is not None:
					return f'T: {self.operation.name}'
				return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

		TO = math_system.TransformOperation
		match self.operation:
			case TO.ConvertIdxUnit:
				row = layout.row(align=True)
				row.prop(self, self.blfields['active_dim'], text='')
				row.prop(self, self.blfields['active_new_unit'], text='')

			case TO.FreqToVacWL:
				row = layout.row(align=True)
				row.prop(self, self.blfields['active_dim'], text='')
				row.prop(self, self.blfields['active_new_unit'], text='')

			case TO.VacWLToFreq:
				row = layout.row(align=True)
				row.prop(self, self.blfields['active_dim'], text='')
				row.prop(self, self.blfields['active_new_unit'], text='')

			case TO.SetIdxUnit:
				row = layout.row(align=True)
				row.prop(self, self.blfields['active_dim'], text='')
				row.prop(self, self.blfields['new_name'], text='')

				row = layout.row(align=True)
				row.prop(self, self.blfields['new_physical_type'], text='')
				row.prop(self, self.blfields['active_new_unit'], text='')

			case TO.FirstColToFirstIdx:
				col = layout.column(align=True)
				row = col.row(align=True)
				row.prop(self, self.blfields['new_name'], text='')
				row.prop(self, self.blfields['active_new_unit'], text='')

				row = col.row(align=True)
				row.prop(self, self.blfields['new_physical_type'], text='')

			case TO.FT1D | TO.InvFT1D:
				layout.prop(self, self.blfields['active_dim'], text='')

	####################
	# - Compute: Func / Array
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'operation', 'dim'},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Func, ct.FlowKind.Info},
		},
		output_sockets={'Expr'},
		output_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def compute_func(
		self, props, input_sockets, output_sockets
	) -> ct.FuncFlow | ct.FlowSignal:
		"""Transform the input `InfoFlow` depending on the transform operation."""
		TO = math_system.TransformOperation

		lazy_func = input_sockets['Expr'][ct.FlowKind.Func]
		info = input_sockets['Expr'][ct.FlowKind.Info]
		output_info = output_sockets['Expr']

		has_info = not ct.FlowSignal.check(info)
		has_lazy_func = not ct.FlowSignal.check(lazy_func)
		has_output_info = not ct.FlowSignal.check(output_info)

		operation = props['operation']
		if operation is not None and has_lazy_func and has_info and has_output_info:
			dim = props['dim']
			match operation:
				case TO.FreqToVacWL | TO.VacWLToFreq | TO.FT1D | TO.InvFT1D:
					if dim is not None and info.has_idx_discrete(dim):
						return lazy_func.compose_within(
							operation.jax_func(axis=info.dim_axis(dim)),
							enclosing_func_output=output_info.output,
							supports_jax=True,
						)
					return ct.FlowSignal.FlowPending

				case _:
					return lazy_func.compose_within(
						operation.jax_func(),
						enclosing_func_output=output_info.output,
						supports_jax=True,
					)

		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		# Loaded
		props={'operation', 'dim', 'new_name', 'new_unit', 'new_physical_type'},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Func, ct.FlowKind.Info, ct.FlowKind.Params}
		},
	)
	def compute_info(  # noqa: PLR0911
		self, props: dict, input_sockets: dict
	) -> ct.InfoFlow | typ.Literal[ct.FlowSignal.FlowPending]:
		"""Transform the input `InfoFlow` depending on the transform operation."""
		TO = math_system.TransformOperation
		operation = props['operation']
		info = input_sockets['Expr'][ct.FlowKind.Info]

		has_info = not ct.FlowSignal.check(info)
		if has_info and operation is not None:
			# Retrieve Properties
			dim = props['dim']
			new_name = props['new_name']
			new_unit = props['new_unit']
			new_physical_type = props['new_physical_type']

			# Retrieve Expression Data
			lazy_func = input_sockets['Expr'][ct.FlowKind.Func]
			params = input_sockets['Expr'][ct.FlowKind.Params]

			has_lazy_func = not ct.FlowSignal.check(lazy_func)
			has_params = not ct.FlowSignal.check(lazy_func)

			# Match Pattern by Operation
			match operation:
				# Covariant Transform
				## -> Needs: Dim, Unit
				case TO.ConvertIdxUnit if dim is not None and new_unit is not None:
					physical_type = spux.PhysicalType.from_unit(dim.unit, optional=True)
					if (
						physical_type is not None
						and new_unit in physical_type.valid_units
					):
						return operation.transform_info(info, dim=dim, unit=new_unit)
					return ct.FlowSignal.FlowPending

				case TO.FreqToVacWL if dim is not None and new_unit is not None and new_unit in spux.PhysicalType.Length.valid_units:
					return operation.transform_info(info, dim=dim, unit=new_unit)

				case TO.VacWLToFreq if dim is not None and new_unit is not None and new_unit in spux.PhysicalType.Freq.valid_units:
					return operation.transform_info(info, dim=dim, unit=new_unit)

				## -> Needs: Dim, Unit, Physical Type
				case TO.SetIdxUnit if (
					dim is not None
					and new_physical_type is not None
					and new_unit in new_physical_type.valid_units
				):
					return operation.transform_info(
						info,
						dim=dim,
						new_dim_name=new_name,
						unit=new_unit,
						physical_type=new_physical_type,
					)

				## -> Needs: Data Column, Name, Unit, Physical Type
				## -> We have to evaluate the lazy function at this point.
				## -> It's the only way to get at the column's data.
				case TO.FirstColToFirstIdx if (
					has_lazy_func
					and has_params
					and not params.symbols
					and new_name is not None
					and new_physical_type is not None
					and new_unit in new_physical_type.valid_units
				):
					data = lazy_func.realize(params)
					if data.shape is not None and len(data.shape) == 2:  # noqa: PLR2004
						data_col = data[:, 0]
						return operation.transform_info(
							info,
							new_dim_name=new_name,
							data_col=data_col,
							unit=new_unit,
							physical_type=new_physical_type,
						)

				# Fold
				## -> Needs: Nothing
				case TO.IntDimToComplex | TO.DimToVec | TO.DimsToMat:
					return operation.transform_info(info)

				# Fourier
				## -> Needs: Dimension
				case TO.FT1D | TO.InvFT1D if dim is not None:
					return operation.transform_info(info, dim=dim)

		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		# Loaded
		props={'operation'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Params},
	)
	def compute_params(self, props, input_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		operation = props['operation']
		params = input_sockets['Expr']

		has_params = not ct.FlowSignal.check(params)
		if has_params and operation is not None:
			return params

		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	TransformMathNode,
]
BL_NODES = {ct.NodeType.TransformMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
