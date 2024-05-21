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
from pathlib import Path

import bpy
import sympy as sp
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


####################
# - Node
####################
class DataFileImporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.DataFileImporter
	bl_label = 'Data File Importer'

	input_sockets: typ.ClassVar = {
		'File Path': sockets.FilePathSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.Func),
	}

	####################
	# - Properties
	####################
	@events.on_value_changed(
		socket_name={'File Path'},
		input_sockets={'File Path'},
		input_socket_kinds={'File Path': ct.FlowKind.Value},
		input_sockets_optional={'File Path': True},
	)
	def on_input_exprs_changed(self, input_sockets) -> None:  # noqa: D102
		has_file_path = not ct.FlowSignal.check(input_sockets['File Path'])

		if has_file_path:
			self.file_path = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property()
	def file_path(self) -> Path:
		"""Retrieve the input file path."""
		file_path = self._compute_input(
			'File Path', kind=ct.FlowKind.Value, optional=True
		)
		has_file_path = not ct.FlowSignal.check(file_path)
		if has_file_path:
			return file_path

		return None

	@bl_cache.cached_bl_property(depends_on={'file_path'})
	def data_file_format(self) -> ct.DataFileFormat | None:
		"""Retrieve the file extension by concatenating all suffixes."""
		if self.file_path is not None:
			return ct.DataFileFormat.from_path(self.file_path)
		return None

	####################
	# - Output Info
	####################
	@bl_cache.cached_bl_property(depends_on={'file_path'})
	def expr_info(self) -> ct.InfoFlow | None:
		"""Retrieve the output expression's `InfoFlow`."""
		info = self.compute_output('Expr', kind=ct.FlowKind.Info)
		has_info = not ct.FlowSignal.check(info)
		if has_info:
			return info
		return None

	####################
	# - UI
	####################
	def draw_label(self):
		"""Show the extracted file name (w/extension) in the node's header label.

		Notes:
			Called by Blender to determine the text to place in the node's header.
		"""
		if self.file_path is not None:
			return 'Load: ' + self.file_path.name

		return self.bl_label

	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Show information about the loaded file."""
		if self.data_file_format is not None:
			box = layout.box()
			row = box.row()
			row.alignment = 'CENTER'
			row.label(text='Data File')

			row = box.row()
			row.alignment = 'CENTER'
			row.label(text=self.file_path.name)

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		pass

	####################
	# - FlowKind.Array|Func
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Func,
		input_sockets={'File Path'},
	)
	def compute_func(self, input_sockets: dict) -> td.Simulation:
		"""Declare a lazy, composable function that returns the loaded data.

		Returns:
			A completely empty `ParamsFlow`, ready to be composed.
		"""
		file_path = input_sockets['File Path']

		has_file_path = not ct.FlowSignal.check(input_sockets['File Path'])

		if has_file_path:
			data_file_format = ct.DataFileFormat.from_path(file_path)
			if data_file_format is not None:
				# Jax Compatibility: Lazy Data Loading
				## -> Delay loading of data from file as long as we can.
				if data_file_format.loader_is_jax_compatible:
					return ct.FuncFlow(
						func=lambda: data_file_format.loader(file_path),
						supports_jax=True,
					)

				# No Jax Compatibility: Eager Data Loading
				## -> Load the data now and bind it.
				data = data_file_format.loader(file_path)
				return ct.FuncFlow(func=lambda: data, supports_jax=True)
			return ct.FlowSignal.FlowPending
		return ct.FlowSignal.FlowPending

	####################
	# - FlowKind.Params|Info
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Params,
	)
	def compute_params(self) -> ct.ParamsFlow:
		"""Declare an empty `Data:Params`, to indicate the start of a function-composition pipeline.

		Returns:
			A completely empty `ParamsFlow`, ready to be composed.
		"""
		return ct.ParamsFlow()

	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Info,
		output_sockets={'Expr'},
		output_socket_kinds={'Expr': ct.FlowKind.Func},
	)
	def compute_info(self, output_sockets) -> ct.InfoFlow:
		"""Declare an `InfoFlow` based on the data shape.

		This currently requires computing the data.
		Note, however, that the incremental cache causes this computation only to happen once when a file is selected.

		Returns:
			A completely empty `ParamsFlow`, ready to be composed.
		"""
		expr = output_sockets['Expr']

		has_expr_func = not ct.FlowSignal.check(expr)

		if has_expr_func:
			data = expr.func_jax()

			# Deduce Dimensionality
			_shape = data.shape
			shape = _shape if _shape is not None else ()
			dim_names = [f'a{i}' for i in range(len(shape))]

			# Return InfoFlow
			## -> TODO: How to interpret the data should be user-defined.
			## -> -- This may require those nice dynamic symbols.
			return ct.InfoFlow(
				dim_names=dim_names,  ## TODO: User
				dim_idx={
					dim_name: ct.RangeFlow(
						start=sp.S(0),  ## TODO: User
						stop=sp.S(shape[i] - 1),  ## TODO: User
						steps=shape[dim_names.index(dim_name)],
						unit=None,  ## TODO: User
					)
					for i, dim_name in enumerate(dim_names)
				},
				output_name='_',
				output_shape=None,
				output_mathtype=spux.MathType.Real,  ## TODO: User
				output_unit=None,  ## TODO: User
			)
		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	DataFileImporterNode,
]
BL_NODES = {
	ct.NodeType.DataFileImporter: (ct.NodeCategory.MAXWELLSIM_INPUTS_FILEIMPORTERS)
}
