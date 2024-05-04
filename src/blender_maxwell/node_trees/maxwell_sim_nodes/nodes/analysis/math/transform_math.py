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
import jax

from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import sockets
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
		'Data': sockets.DataSocketDef(format='jax'),
	}
	input_socket_sets: typ.ClassVar = {
		'Fourier': {},
		'Affine': {},
		'Convolve': {},
	}
	output_sockets: typ.ClassVar = {
		'Data': sockets.DataSocketDef(format='jax'),
	}

	####################
	# - Properties
	####################
	operation: enum.Enum = bl_cache.BLField(
		prop_ui=True, enum_cb=lambda self, _: self.search_operations()
	)

	def search_operations(self) -> list[ct.BLEnumElement]:
		if self.active_socket_set == 'Fourier':  # noqa: SIM114
			items = []
		elif self.active_socket_set == 'Affine':  # noqa: SIM114
			items = []
		elif self.active_socket_set == 'Convolve':
			items = []
		else:
			msg = f'Active socket set {self.active_socket_set} is unknown'
			raise RuntimeError(msg)

		return [(*item, '', i) for i, item in enumerate(items)]

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['operation'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name='active_socket_set',
	)
	def on_socket_set_changed(self):
		self.operation = bl_cache.Signal.ResetEnumItems

	####################
	# - Compute: LazyValueFunc / Array
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.LazyValueFunc,
		props={'active_socket_set', 'operation'},
		input_sockets={'Data'},
		input_socket_kinds={
			'Data': ct.FlowKind.LazyValueFunc,
		},
	)
	def compute_data(self, props: dict, input_sockets: dict):
		has_data = not ct.FlowSignal.check(input_sockets['Data'])
		if not has_data or props['operation'] == 'NONE':
			return ct.FlowSignal.FlowPending

		mapping_func: typ.Callable[[jax.Array], jax.Array] = {
			'Fourier': {},
			'Affine': {},
			'Convolve': {},
		}[props['active_socket_set']][props['operation']]

		# Compose w/Lazy Root Function Data
		return input_sockets['Data'].compose_within(
			mapping_func,
			supports_jax=True,
		)

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Array,
		output_sockets={'Data'},
		output_socket_kinds={
			'Data': {ct.FlowKind.LazyValueFunc, ct.FlowKind.Params},
		},
	)
	def compute_array(self, output_sockets: dict) -> ct.ArrayFlow:
		lazy_value_func = output_sockets['Data'][ct.FlowKind.LazyValueFunc]
		params = output_sockets['Data'][ct.FlowKind.Params]

		if all(not ct.FlowSignal.check(inp) for inp in [lazy_value_func, params]):
			return ct.ArrayFlow(
				values=lazy_value_func.func_jax(
					*params.func_args, **params.func_kwargs
				),
				unit=None,
			)

		return ct.FlowSignal.FlowPending

	####################
	# - Compute Auxiliary: Info / Params
	####################
	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Info,
		props={'active_socket_set', 'operation'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Info},
	)
	def compute_data_info(self, props: dict, input_sockets: dict) -> ct.InfoFlow:
		info = input_sockets['Data']
		if ct.FlowSignal.check(info):
			return ct.FlowSignal.FlowPending

		return info

	@events.computes_output_socket(
		'Data',
		kind=ct.FlowKind.Params,
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Params},
	)
	def compute_data_params(self, input_sockets: dict) -> ct.ParamsFlow | ct.FlowSignal:
		return input_sockets['Data']


####################
# - Blender Registration
####################
BL_REGISTER = [
	TransformMathNode,
]
BL_NODES = {ct.NodeType.TransformMath: (ct.NodeCategory.MAXWELLSIM_ANALYSIS_MATH)}
