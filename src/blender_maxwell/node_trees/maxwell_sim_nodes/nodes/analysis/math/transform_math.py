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
import jax.numpy as jnp
import jaxtyping as jtyp
import sympy as sp
import sympy.physics.units as spu

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

	# Fold
	IntDimToComplex = enum.auto()
	DimToVec = enum.auto()
	DimsToMat = enum.auto()

	# Fourier
	FFT1D = enum.auto()
	InvFFT1D = enum.auto()

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
			# Fold
			TO.IntDimToComplex: '→ ℂ',
			TO.DimToVec: '→ Vector',
			TO.DimsToMat: '→ Matrix',
			# Fourier
			TO.FFT1D: 't → 𝑓',
			TO.InvFFT1D: '𝑓 → t',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
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
	# - Ops from Shape
	####################
	@staticmethod
	def by_element_shape(info: ct.InfoFlow) -> list[typ.Self]:
		TO = TransformOperation
		operations = []

		# Covariant Transform
		## Freq <-> VacWL
		for dim_name in info.dim_names:
			if info.dim_physical_types[dim_name] == spux.PhysicalType.Freq:
				operations.append(TO.FreqToVacWL)

			if info.dim_physical_types[dim_name] == spux.PhysicalType.Freq:
				operations.append(TO.VacWLToFreq)

		# Fold
		## (Last) Int Dim (=2) to Complex
		if len(info.dim_names) >= 1:
			last_dim_name = info.dim_names[-1]
			if info.dim_lens[last_dim_name] == 2:  # noqa: PLR2004
				operations.append(TO.IntDimToComplex)

		## To Vector
		if len(info.dim_names) >= 1:
			operations.append(TO.DimToVec)

		## To Matrix
		if len(info.dim_names) >= 2:  # noqa: PLR2004
			operations.append(TO.DimsToMat)

		# Fourier
		## 1D Fourier
		if info.dim_names:
			last_physical_type = info.dim_physical_types[info.dim_names[-1]]
			if last_physical_type == spux.PhysicalType.Time:
				operations.append(TO.FFT1D)
			if last_physical_type == spux.PhysicalType.Freq:
				operations.append(TO.InvFFT1D)

		return operations

	####################
	# - Function Properties
	####################
	@property
	def sp_func(self):
		TO = TransformOperation
		return {
			# Covariant Transform
			TO.FreqToVacWL: lambda expr: expr,
			TO.VacWLToFreq: lambda expr: expr,
			# Fold
			# TO.IntDimToComplex: lambda expr: expr,  ## TODO: Won't work?
			TO.DimToVec: lambda expr: expr,
			TO.DimsToMat: lambda expr: expr,
			# Fourier
			TO.FFT1D: lambda expr: sp.fourier_transform(
				expr, sim_symbols.t, sim_symbols.freq
			),
			TO.InvFFT1D: lambda expr: sp.fourier_transform(
				expr, sim_symbols.freq, sim_symbols.t
			),
		}[self]

	@property
	def jax_func(self):
		TO = TransformOperation
		return {
			# Covariant Transform
			TO.FreqToVacWL: lambda expr: expr,
			TO.VacWLToFreq: lambda expr: expr,
			# Fold
			## -> To Complex: With a little imagination, this is a noop :)
			## -> **Requires** dims[-1] to be integer-indexed w/length of 2.
			TO.IntDimToComplex: lambda expr: expr.view(dtype=jnp.complex64).squeeze(),
			TO.DimToVec: lambda expr: expr,
			TO.DimsToMat: lambda expr: expr,
			# Fourier
			TO.FFT1D: lambda expr: jnp.fft(expr),
			TO.InvFFT1D: lambda expr: jnp.ifft(expr),
		}[self]

	def transform_info(
		self,
		info: ct.InfoFlow | None,
		data: jtyp.Shaped[jtyp.Array, '...'] | None = None,
		unit: spux.Unit | None = None,
	) -> ct.InfoFlow | None:
		TO = TransformOperation
		if not info.dim_names:
			return None
		return {
			# Index
			TO.FreqToVacWL: lambda: info.replace_dim(
				(f_dim := info.dim_names[-1]),
				[
					'wl',
					info.dim_idx[f_dim].rescale(
						lambda el: sci_constants.vac_speed_of_light / el,
						reverse=True,
						new_unit=spu.nanometer,
					),
				],
			),
			TO.VacWLToFreq: lambda: info.replace_dim(
				(wl_dim := info.dim_names[-1]),
				[
					'f',
					info.dim_idx[wl_dim].rescale(
						lambda el: sci_constants.vac_speed_of_light / el,
						reverse=True,
						new_unit=spux.THz,
					),
				],
			),
			# Fold
			TO.IntDimToComplex: lambda: info.delete_dimension(
				info.dim_names[-1]
			).set_output_mathtype(spux.MathType.Complex),
			TO.DimToVec: lambda: info.shift_last_input,
			TO.DimsToMat: lambda: info.shift_last_input.shift_last_input,
			# Fourier
			TO.FFT1D: lambda: info.replace_dim(
				info.dim_names[-1],
				[
					'f',
					ct.RangeFlow(start=0, stop=sp.oo, steps=0, unit=spu.hertz),
				],
			),
			TO.InvFFT1D: info.replace_dim(
				info.dim_names[-1],
				[
					't',
					ct.RangeFlow(start=0, stop=sp.oo, steps=0, unit=spu.second),
				],
			),
		}.get(self, lambda: info)()


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
	operation: TransformOperation = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_operations(),
		cb_depends_on={'expr_info'},
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.expr_info is not None:
			return [
				operation.bl_enum_element(i)
				for i, operation in enumerate(
					TransformOperation.by_element_shape(self.expr_info)
				)
			]
		return []

	####################
	# - UI
	####################
	def draw_label(self):
		if self.operation is not None:
			return 'Transform: ' + TransformOperation.to_name(self.operation)

		return self.bl_label

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - Compute: LazyValueFunc / Array
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Value,
		props={'operation'},
		input_sockets={'Expr'},
	)
	def compute_value(self, props, input_sockets) -> ct.ValueFlow | ct.FlowSignal:
		operation = props['operation']
		expr = input_sockets['Expr']

		has_expr_value = not ct.FlowSignal.check(expr)

		# Compute Sympy Function
		## -> The operation enum directly provides the appropriate function.
		if has_expr_value and operation is not None:
			return operation.sp_func(expr)

		return ct.Flowsignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.LazyValueFunc,
		props={'operation'},
		input_sockets={'Expr'},
		input_socket_kinds={
			'Expr': ct.FlowKind.LazyValueFunc,
		},
	)
	def compute_func(
		self, props, input_sockets
	) -> ct.LazyValueFuncFlow | ct.FlowSignal:
		operation = props['operation']
		expr = input_sockets['Expr']

		has_expr = not ct.FlowSignal.check(expr)

		if has_expr and operation is not None:
			return expr.compose_within(
				operation.jax_func,
				supports_jax=True,
			)
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Info|Params
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		props={'operation'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def compute_info(
		self, props: dict, input_sockets: dict
	) -> ct.InfoFlow | typ.Literal[ct.FlowSignal.FlowPending]:
		operation = props['operation']
		info = input_sockets['Expr']

		has_info = not ct.FlowSignal.check(info)

		if has_info and operation is not None:
			transformed_info = operation.transform_info(info)

			if transformed_info is None:
				return ct.FlowSignal.FlowPending
			return transformed_info

		return ct.FlowSignal.FlowPending

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Params},
	)
	def compute_params(self, input_sockets: dict) -> ct.ParamsFlow | ct.FlowSignal:
		has_params = not ct.FlowSignal.check(input_sockets['Expr'])
		if has_params:
			return input_sockets['Expr']
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	TransformMathNode,
]
BL_NODES = {ct.NodeType.TransformMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
