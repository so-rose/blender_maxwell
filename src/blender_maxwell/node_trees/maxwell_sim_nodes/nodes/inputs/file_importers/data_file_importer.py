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
from pathlib import Path

import bpy
import sympy as sp
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger, sim_symbols
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
		# Trigger
		socket_name={'File Path'},
		# Loaded
		input_sockets={'File Path'},
		input_socket_kinds={'File Path': ct.FlowKind.Value},
		input_sockets_optional={'File Path': True},
		# Flow
		## -> See docs in TransformMathNode
		stop_propagation=True,
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
	@bl_cache.cached_bl_property(
		depends_on={
			'output_name',
			'output_mathtype',
			'output_physical_type',
			'output_unit',
		}
		| {f'dim_{i}_name' for i in range(6)}
	)
	def expr_info(self) -> ct.InfoFlow | None:
		"""Retrieve the output expression's `InfoFlow`."""
		info = self.compute_output('Expr', kind=ct.FlowKind.Info)
		has_info = not ct.FlowSignal.check(info)
		if has_info:
			return info
		return None

	####################
	# - Info Guides
	####################
	output_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.Data
	)
	output_mathtype: spux.MathType = bl_cache.BLField(spux.MathType.Real)
	output_physical_type: spux.PhysicalType = bl_cache.BLField(
		spux.PhysicalType.NonPhysical
	)
	output_unit: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_units(self.output_physical_type),
		cb_depends_on={'output_physical_type'},
	)

	dim_0_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.LowerA
	)
	dim_1_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.LowerB
	)
	dim_2_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.LowerC
	)
	dim_3_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.LowerD
	)
	dim_4_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.LowerE
	)
	dim_5_name: sim_symbols.SimSymbolName = bl_cache.BLField(
		sim_symbols.SimSymbolName.LowerF
	)

	def search_units(self, physical_type: spux.PhysicalType) -> list[ct.BLEnumElement]:
		if physical_type is not spux.PhysicalType.NonPhysical:
			return [
				(sp.sstr(unit), spux.sp_to_str(unit), sp.sstr(unit), '', i)
				for i, unit in enumerate(physical_type.valid_units)
			]
		return []

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
		"""Draw loaded properties."""
		col = layout.column(align=True)
		if self.expr_info is not None:
			for i in range(len(self.expr_info.dims)):
				col.prop(self, self.blfields[f'dim_{i}_name'], text=f'Dim {i}')

			row = col.row(align=True)
			row.alignment = 'CENTER'
			row.label(text='Output')

			row = col.row(align=True)
			row.prop(self, self.blfields['output_name'], text='')
			row.prop(self, self.blfields['output_mathtype'], text='')
			row = col.row(align=True)
			row.prop(self, self.blfields['output_physical_type'], text='')
			row.prop(self, self.blfields['output_unit'], text='')

	####################
	# - FlowKind.Array|Func
	####################
	@events.computes_output_socket(
		'Expr',
		kind=ct.FlowKind.Func,
		# Loaded
		props={'output_name', 'output_mathtype', 'output_physical_type', 'output_unit'},
		input_sockets={'File Path'},
	)
	def compute_func(self, props, input_sockets) -> td.Simulation:
		"""Declare a lazy, composable function that returns the loaded data.

		Returns:
			A completely empty `ParamsFlow`, ready to be composed.
		"""
		file_path = input_sockets['File Path']
		has_file_path = not ct.FlowSignal.check(file_path)

		func_output = sim_symbols.SimSymbol(
			sym_name=props['output_name'],
			mathtype=props['output_mathtype'],
			physical_type=props['output_physical_type'],
			unit=props['output_unit'],
		)
		if has_file_path and file_path is not None:
			data_file_format = ct.DataFileFormat.from_path(file_path)
			if data_file_format is not None:
				# Jax Compatibility: Lazy Data Loading
				## -> Delay loading of data from file as long as we can.
				if data_file_format.loader_is_jax_compatible:
					return ct.FuncFlow(
						func=lambda: data_file_format.loader(file_path),
						func_output=func_output,
						supports_jax=True,
					)

				# No Jax Compatibility: Eager Data Loading
				## -> Load the data now and bind it.
				data = data_file_format.loader(file_path)
				return ct.FuncFlow(
					func=lambda: data,
					func_output=func_output,
					supports_jax=True,
				)
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
		# Loaded
		props={'output_name', 'output_mathtype', 'output_physical_type', 'output_unit'}
		| {f'dim_{i}_name' for i in range(6)},
		output_sockets={'Expr'},
		output_socket_kinds={'Expr': ct.FlowKind.Func},
	)
	def compute_info(self, props, output_sockets) -> ct.InfoFlow:
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

			# Deduce Dimension Symbols
			## -> They are all chronically integer indices.
			## -> The FilterNode can be used to "steal" an index from the data.
			shape = data.shape if data.shape is not None else ()
			dims = {
				sim_symbols.idx(None).update(
					sym_name=props[f'dim_{i}_name'],
					interval_finite_z=(0, elements),
					interval_inf=(False, False),
					interval_closed=(True, True),
				): [str(j) for j in range(elements)]
				for i, elements in enumerate(shape)
			}

			return ct.InfoFlow(
				dims=dims,
				output=sim_symbols.SimSymbol(
					sym_name=props['output_name'],
					mathtype=props['output_mathtype'],
					physical_type=props['output_physical_type'],
					unit=props['output_unit'],
				),
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
