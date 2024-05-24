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
import functools
import typing as typ

import bpy
import jax.numpy as jnp
import jaxtyping as jtyp
import sympy as sp

from blender_maxwell.utils import bl_cache, logger, sci_constants, sim_symbols
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


####################
# - Operation Enum
####################
class TransformOperation(enum.StrEnum):
	"""Valid operations for the `MapMathNode`.

	Attributes:
		FreqToVacWL: Transform frequency axes to be indexed by vacuum wavelength.
		VacWLToFreq: Transform vacuum wavelength axes to be indexed by frequency.
		FFT: Compute the fourier transform of the input expression.
		InvFFT: Compute the inverse fourier transform of the input expression.
	"""

	# Covariant Transform
	FreqToVacWL = enum.auto()
	VacWLToFreq = enum.auto()
	ConvertIdxUnit = enum.auto()
	SetIdxUnit = enum.auto()
	FirstColToFirstIdx = enum.auto()

	# Fold
	IntDimToComplex = enum.auto()
	DimToVec = enum.auto()
	DimsToMat = enum.auto()

	# Fourier
	FT1D = enum.auto()
	InvFT1D = enum.auto()

	# TODO: Affine
	## TODO

	####################
	# - UI
	####################
	@staticmethod
	def to_name(value: typ.Self) -> str:
		TO = TransformOperation
		return {
			# Covariant Transform
			TO.FreqToVacWL: '𝑓 → λᵥ',
			TO.VacWLToFreq: 'λᵥ → 𝑓',
			TO.ConvertIdxUnit: 'Convert Dim',
			TO.SetIdxUnit: 'Set Dim',
			TO.FirstColToFirstIdx: '1st Col → Dim',
			# Fold
			TO.IntDimToComplex: '→ ℂ',
			TO.DimToVec: '→ Vector',
			TO.DimsToMat: '→ Matrix',
			## TODO: Vector to new last-dim integer
			## TODO: Matrix to two last-dim integers
			# Fourier
			TO.FT1D: '→ 𝑓',
			TO.InvFT1D: '𝑓 →',
		}[value]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		TO = TransformOperation
		return (
			str(self),
			TO.to_name(self),
			TO.to_name(self),
			TO.to_icon(self),
			i,
		)

	####################
	# - Methods
	####################
	@property
	def num_dim_inputs(self) -> None:
		"""The number of axes that should be passed as inputs to `func_jax` when evaluating it.

		Especially useful for `ParamFlow`, when deciding whether to pass an integer-axis argument based on a user-selected dimension.
		"""
		TO = TransformOperation
		return {
			# Covariant Transform
			TO.FreqToVacWL: 1,
			TO.VacWLToFreq: 1,
			TO.ConvertIdxUnit: 1,
			TO.SetIdxUnit: 1,
			TO.FirstColToFirstIdx: 0,
			# Fold
			TO.IntDimToComplex: 0,
			TO.DimToVec: 0,
			TO.DimsToMat: 0,
			## TODO: Vector to new last-dim integer
			## TODO: Matrix to two last-dim integers
			# Fourier
			TO.FT1D: 1,
			TO.InvFT1D: 1,
		}[self]

	def valid_dims(self, info: ct.InfoFlow) -> list[typ.Self]:
		TO = TransformOperation
		match self:
			case TO.FreqToVacWL | TO.FT1D:
				return [
					dim
					for dim in info.dims
					if dim.physical_type is spux.PhysicalType.Freq
				]

			case TO.VacWLToFreq | TO.InvFT1D:
				return [
					dim
					for dim in info.dims
					if dim.physical_type is spux.PhysicalType.Length
				]

			case TO.ConvertIdxUnit | TO.SetIdxUnit:
				return [dim for dim in info.dims if not info.has_idx_labels(dim)]

			## ColDimToComplex: Implicit Last Dimension
			## DimToVec: Implicit Last Dimension
			## DimsToMat: Implicit Last 2 Dimensions

			case TO.FT1D | TO.InvFT1D:
				# Filter by Axis Uniformity
				## -> FT requires uniform axis (aka. must be RangeFlow).
				## -> NOTE: If FT isn't popping up, check ExtractDataNode.
				return [dim for dim in info.dims if info.is_idx_uniform(dim)]

		return []

	@staticmethod
	def by_info(info: ct.InfoFlow) -> list[typ.Self]:
		TO = TransformOperation
		operations = []

		# Covariant Transform
		## Freq -> VacWL
		if TO.FreqToVacWL.valid_dims(info):
			operations += [TO.FreqToVacWL]

		## VacWL -> Freq
		if TO.VacWLToFreq.valid_dims(info):
			operations += [TO.VacWLToFreq]

		## Convert Index Unit
		if TO.ConvertIdxUnit.valid_dims(info):
			operations += [TO.ConvertIdxUnit]

		if TO.SetIdxUnit.valid_dims(info):
			operations += [TO.SetIdxUnit]

		## Column to First Index (Array)
		if (
			len(info.dims) == 2  # noqa: PLR2004
			and info.first_dim.mathtype is spux.MathType.Integer
			and info.last_dim.mathtype is spux.MathType.Integer
			and info.output.shape_len == 0
		):
			operations += [TO.FirstColToFirstIdx]

		# Fold
		## Last Dim -> Complex
		if (
			info.dims
			# Output is Int|Rat|Real
			and (
				info.output.mathtype
				in [spux.MathType.Integer, spux.MathType.Rational, spux.MathType.Real]
			)
			# Last Axis is Integer of Length 2
			and info.last_dim.mathtype is spux.MathType.Integer
			and info.has_idx_labels(info.last_dim)
			and len(info.dims[info.last_dim]) == 2  # noqa: PLR2004
		):
			operations += [TO.IntDimToComplex]

		## Last Dim -> Vector
		if len(info.dims) >= 1 and info.output.shape_len == 0:
			operations += [TO.DimToVec]

		## Last Dim -> Matrix
		if len(info.dims) >= 2 and info.output.shape_len == 0:  # noqa: PLR2004
			operations += [TO.DimsToMat]

		# Fourier
		if TO.FT1D.valid_dims(info):
			operations += [TO.FT1D]

		if TO.InvFT1D.valid_dims(info):
			operations += [TO.InvFT1D]

		return operations

	####################
	# - Function Properties
	####################
	@functools.cached_property
	def jax_func(self):
		TO = TransformOperation
		return {
			# Covariant Transform
			## -> Freq <-> WL is a rescale (noop) AND flip (not noop).
			TO.FreqToVacWL: lambda expr, axis: jnp.flip(expr, axis=axis),
			TO.VacWLToFreq: lambda expr, axis: jnp.flip(expr, axis=axis),
			TO.ConvertIdxUnit: lambda expr: expr,
			TO.SetIdxUnit: lambda expr: expr,
			TO.FirstColToFirstIdx: lambda expr: jnp.delete(expr, 0, axis=1),
			# Fold
			## -> To Complex: This should generally be a no-op.
			TO.IntDimToComplex: lambda expr: jnp.squeeze(
				expr.view(dtype=jnp.complex64), axis=-1
			),
			TO.DimToVec: lambda expr: expr,
			TO.DimsToMat: lambda expr: expr,
			# Fourier
			TO.FT1D: lambda expr, axis: jnp.fft(expr, axis=axis),
			TO.InvFT1D: lambda expr, axis: jnp.ifft(expr, axis=axis),
		}[self]

	def transform_info(
		self,
		info: ct.InfoFlow,
		dim: sim_symbols.SimSymbol | None = None,
		data_col: jtyp.Shaped[jtyp.Array, ' size'] | None = None,
		new_dim_name: str | None = None,
		unit: spux.Unit | None = None,
		physical_type: spux.PhysicalType | None = None,
	) -> ct.InfoFlow:
		TO = TransformOperation
		return {
			# Covariant Transform
			TO.FreqToVacWL: lambda: info.replace_dim(
				(f_dim := dim),
				[
					sim_symbols.wl(unit),
					info.dims[f_dim].rescale(
						lambda el: sci_constants.vac_speed_of_light / el,
						reverse=True,
						new_unit=unit,
					),
				],
			),
			TO.VacWLToFreq: lambda: info.replace_dim(
				(wl_dim := dim),
				[
					sim_symbols.freq(unit),
					info.dims[wl_dim].rescale(
						lambda el: sci_constants.vac_speed_of_light / el,
						reverse=True,
						new_unit=unit,
					),
				],
			),
			TO.ConvertIdxUnit: lambda: info.replace_dim(
				dim,
				dim.update(unit=unit),
				(
					info.dims[dim].rescale_to_unit(unit)
					if info.has_idx_discrete(dim)
					else None  ## Continuous -- dim SimSymbol already scaled
				),
			),
			TO.SetIdxUnit: lambda: info.replace_dim(
				dim,
				dim.update(
					sym_name=new_dim_name, physical_type=physical_type, unit=unit
				),
				(
					info.dims[dim].correct_unit(unit)
					if info.has_idx_discrete(dim)
					else None  ## Continuous -- dim SimSymbol already scaled
				),
			),
			TO.FirstColToFirstIdx: lambda: info.replace_dim(
				info.first_dim,
				info.first_dim.update(
					mathtype=spux.MathType.from_jax_array(data_col),
					unit=unit,
				),
				ct.ArrayFlow(values=data_col, unit=unit),
			).slice_dim(info.last_dim, (1, len(info.dims[info.last_dim]), 1)),
			# Fold
			TO.IntDimToComplex: lambda: info.delete_dim(info.last_dim).update_output(
				mathtype=spux.MathType.Complex
			),
			TO.DimToVec: lambda: info.fold_last_input(),
			TO.DimsToMat: lambda: info.fold_last_input().fold_last_input(),
			# Fourier
			TO.FT1D: lambda: info.replace_dim(
				dim,
				[
					# FT'ed Unit: Reciprocal of the Original Unit
					dim.update(
						unit=1 / dim.unit if dim.unit is not None else 1
					),  ## TODO: Okay to not scale interval?
					# FT'ed Bounds: Reciprocal of the Original Unit
					info.dims[dim].bound_fourier_transform,
				],
			),
			TO.InvFT1D: lambda: info.replace_dim(
				info.last_dim,
				[
					# FT'ed Unit: Reciprocal of the Original Unit
					dim.update(
						unit=1 / dim.unit if dim.unit is not None else 1
					),  ## TODO: Okay to not scale interval?
					# FT'ed Bounds: Reciprocal of the Original Unit
					## -> Note the midpoint may revert to 0.
					## -> See docs for `RangeFlow.bound_inv_fourier_transform` for more.
					info.dims[dim].bound_inv_fourier_transform,
				],
			),
		}[self]()


####################
# - Node
####################
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
	operation: TransformOperation = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_info'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.expr_info is not None:
			return [
				operation.bl_enum_element(i)
				for i, operation in enumerate(
					TransformOperation.by_info(self.expr_info)
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
			return self.expr_info.dim_by_name(self.active_dim)
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
		cb_depends_on={'dim', 'new_physical_type'},
	)

	def search_units(self) -> list[ct.BLEnumElement]:
		if self.dim is not None:
			if self.dim.physical_type is not spux.PhysicalType.NonPhysical:
				unit_name = sp.sstr(self.dim.unit)
				return [
					(
						sp.sstr(unit),
						spux.sp_to_str(unit),
						sp.sstr(unit),
						'',
						0,
					)
					for unit in self.dim.physical_type.valid_units
				]

			if self.dim.unit is not None:
				unit_name = sp.sstr(self.dim.unit)
				return [
					(
						unit_name,
						spux.sp_to_str(self.dim.unit),
						unit_name,
						'',
						0,
					)
				]
		if self.new_physical_type is not spux.PhysicalType.NonPhysical:
			return [
				(
					sp.sstr(unit),
					spux.sp_to_str(unit),
					sp.sstr(unit),
					'',
					i,
				)
				for i, unit in enumerate(self.new_physical_type.valid_units)
			]

		return []

	@bl_cache.cached_bl_property(depends_on={'active_new_unit'})
	def new_unit(self) -> spux.Unit:
		if self.active_new_unit is not None:
			return spux.unit_str_to_unit(self.active_new_unit)

		return None

	####################
	# - UI
	####################
	def draw_label(self):
		if self.operation is not None:
			return 'T: ' + TransformOperation.to_name(self.operation)

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

		if self.operation is not None and self.operation.num_dim_inputs == 1:
			TO = TransformOperation
			layout.prop(self, self.blfields['active_dim'], text='')

			if self.operation in [TO.ConvertIdxUnit, TO.SetIdxUnit]:
				col = layout.column(align=True)
				if self.operation is TransformOperation.ConvertIdxUnit:
					col.prop(self, self.blfields['active_new_unit'], text='')

				if self.operation is TransformOperation.SetIdxUnit:
					col.prop(self, self.blfields['new_physical_type'], text='')

					row = col.row(align=True)
					row.prop(self, self.blfields['new_name'], text='')
					row.prop(self, self.blfields['active_new_unit'], text='')

	####################
	# - Compute: Func / Array
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Func,
		props={'operation'},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': ct.FlowKind.Func,
		},
	)
	def compute_func(self, props, input_sockets) -> ct.FuncFlow | ct.FlowSignal:
		operation = props['operation']
		lazy_func = input_sockets['Expr']

		has_lazy_func = not ct.FlowSignal.check(lazy_func)

		if has_lazy_func and operation is not None:
			return lazy_func.compose_within(
				operation.jax_func,
				supports_jax=True,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		props={'operation', 'dim', 'new_name', 'new_unit', 'new_physical_type'},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': {ct.FlowKind.Func, ct.FlowKind.Info, ct.FlowKind.Params}
		},
	)
	def compute_info(
		self, props: dict, input_sockets: dict
	) -> ct.InfoFlow | typ.Literal[ct.FlowSignal.FlowPending]:
		operation = props['operation']
		info = input_sockets['Expr'][ct.FlowKind.Info]

		has_info = not ct.FlowSignal.check(info)

		dim = props['dim']
		new_name = props['new_name']
		new_unit = props['new_unit']
		new_physical_type = props['new_physical_type']
		if has_info and operation is not None:
			# First Column to First Index
			## -> We have to evaluate the lazy function at this point.
			## -> It's the only way to get at the column data.
			if operation is TransformOperation.FirstColToFirstIdx:
				lazy_func = input_sockets['Expr'][ct.FlowKind.Func]
				params = input_sockets['Expr'][ct.FlowKind.Params]
				has_lazy_func = not ct.FlowSignal.check(lazy_func)
				has_params = not ct.FlowSignal.check(lazy_func)

				if has_lazy_func and has_params and not params.symbols:
					data = lazy_func.realize(params)
					if data.shape is not None and len(data.shape) == 2:
						data_col = data[:, 0]
						return operation.transform_info(info, data_col=data_col)
				return ct.FlowSignal.FlowPending

			# Check Not-Yet-Updated Dimension
			## - Operation changes before dimensions.
			## - If InfoFlow is requested in this interim, big problem.
			if dim is None and operation.num_dim_inputs > 0:
				return ct.FlowSignal.FlowPending

			return operation.transform_info(
				info,
				dim=dim,
				new_dim_name=new_name,
				unit=new_unit,
				physical_type=new_physical_type,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		props={'operation', 'dim'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': {ct.FlowKind.Params, ct.FlowKind.Info}},
	)
	def compute_params(self, props, input_sockets) -> ct.ParamsFlow | ct.FlowSignal:
		info = input_sockets['Expr'][ct.FlowKind.Info]
		params = input_sockets['Expr'][ct.FlowKind.Params]

		has_info = not ct.FlowSignal.check(info)
		has_params = not ct.FlowSignal.check(params)

		operation = props['operation']
		dim = props['dim']
		if has_info and has_params and operation is not None:
			# Axis Required: Insert by-Dimension
			## -> Some transformations ex. FT require setting an axis.
			## -> The user selects which dimension the op should be done along.
			## -> This dimension is converted to an axis integer.
			## -> Finally, we pass the argument via params.
			if operation.num_dim_inputs == 1:
				axis = info.dim_axis(dim) if dim is not None else None
				return params.compose_within(enclosing_func_args=[axis])

			return params
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	TransformMathNode,
]
BL_NODES = {ct.NodeType.TransformMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
