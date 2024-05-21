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

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


####################
# - Operators
####################
class ExportDataFile(bpy.types.Operator):
	"""Exports data from the input to `DataFileExporterNode` to the file path given on the same node, if the path is compatible with the chosen export format (a property on the node)."""

	bl_idname = ct.OperatorType.NodeExportDataFile
	bl_label = 'Save Data File'
	bl_description = 'Save a file with the contents, name, and format indicated by a NodeExportDataFile'

	@classmethod
	def poll(cls, context):
		return (
			# Check Node
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and (node := context.node).node_type == ct.NodeType.DataFileExporter
			# Check Expr
			and node.is_file_path_compatible_with_export_format
		)

	def execute(self, context: bpy.types.Context):
		node = context.node

		node.export_format.saver(node.file_path, node.expr_data, node.expr_info)
		return {'FINISHED'}


####################
# - Node
####################
class DataFileExporterNode(base.MaxwellSimNode):
	# """Export input data to a supported
	node_type = ct.NodeType.DataFileExporter
	bl_label = 'Data File Importer'

	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(active_kind=ct.FlowKind.LazyValueFunc),
		'File Path': sockets.FilePathSocketDef(),
	}

	####################
	# - Properties: Expr Info
	####################
	@events.on_value_changed(
		socket_name={'Expr'},
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': ct.FlowKind.Info},
	)
	def on_input_exprs_changed(self, input_sockets) -> None:  # noqa: D102
		has_expr = not ct.FlowSignal.check(input_sockets['Expr'])

		if has_expr:
			self.expr_info = bl_cache.Signal.InvalidateCache

	@bl_cache.cached_bl_property(depends_on={'file_path'})
	def expr_info(self) -> ct.InfoFlow | None:
		"""Retrieve the input expression's `InfoFlow`."""
		info = self._compute_input('Expr', kind=ct.FlowKind.Info)
		has_info = not ct.FlowSignal.check(info)
		if has_info:
			return info
		return None

	@property
	def expr_data(self) -> typ.Any | None:
		"""Retrieve the input expression's data by evaluating its `LazyValueFunc`."""
		func = self._compute_input('Expr', kind=ct.FlowKind.LazyValueFunc)
		params = self._compute_input('Expr', kind=ct.FlowKind.Params)

		has_func = not ct.FlowSignal.check(func)
		has_params = not ct.FlowSignal.check(params)
		if has_func and has_params:
			symbol_values = {
				sym.name: self._compute_input(sym.name, kind=ct.FlowKind.Value)
				for sym in params.sorted_symbols
			}
			return func.func_jax(
				*params.scaled_func_args(spux.UNITS_SI, symbol_values=symbol_values),
				**params.scaled_func_kwargs(spux.UNITS_SI, symbol_values=symbol_values),
			)

		return None

	####################
	# - Properties: File Path
	####################
	@events.on_value_changed(
		socket_name={'File Path'},
		input_sockets={'File Path'},
		input_socket_kinds={'File Path': ct.FlowKind.Value},
		input_sockets_optional={'File Path': True},
	)
	def on_file_path_changed(self, input_sockets) -> None:  # noqa: D102
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

	####################
	# - Properties: Export Format
	####################
	export_format: ct.DataFileFormat = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_export_formats(),
		cb_depends_on={'expr_info'},
	)

	def search_export_formats(self):
		if self.expr_info is not None:
			return [
				data_file_format.bl_enum_element(i)
				for i, data_file_format in enumerate(list(ct.DataFileFormat))
				if data_file_format.is_info_compatible(self.expr_info)
			]
		return ct.DataFileFormat.bl_enum_elements()

	####################
	# - Properties: File Path Compatibility
	####################
	@bl_cache.cached_bl_property(depends_on={'file_path', 'export_format'})
	def is_file_path_compatible_with_export_format(self) -> bool | None:
		"""Determine whether the given file path is actually compatible with the desired export format."""
		if self.file_path is not None and self.export_format is not None:
			return self.export_format.is_path_compatible(self.file_path)
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
			return 'Save: ' + self.file_path.name

		return self.bl_label

	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		"""Show information about the loaded file."""
		if self.export_format is not None:
			box = layout.box()
			row = box.row()
			row.alignment = 'CENTER'
			row.label(text='Data File')

			row = box.row()
			row.alignment = 'CENTER'
			row.label(text=self.file_path.name)

			compatibility = self.is_file_path_compatible_with_export_format
			if compatibility is not None:
				row = box.row()
				row.alignment = 'CENTER'
				if compatibility:
					row.label(text='Valid Path | Format', icon='CHECKMARK')
				else:
					row.label(text='Invalid Path | Format', icon='ERROR')

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['export_format'], text='')
		layout.operator(ct.OperatorType.NodeExportDataFile, text='Save Data File')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		socket_name='Expr',
		run_on_init=True,
		# Loaded
		input_sockets={'Expr'},
		input_socket_kinds={'Expr': {ct.FlowKind.Info, ct.FlowKind.Params}},
		input_sockets_optional={'Expr': True},
	)
	def on_expr_changed(self, input_sockets: dict) -> None:
		"""Declare any loose input sockets needed to realize the input expr's symbols."""
		info = input_sockets['Expr'][ct.FlowKind.Info]
		params = input_sockets['Expr'][ct.FlowKind.Params]

		has_info = not ct.FlowSignal.check(info)
		has_params = not ct.FlowSignal.check(params)

		# Provide Sockets for Symbol Realization
		## -> Only happens if Params contains not-yet-realized symbols.
		if has_info and has_params and params.symbols:
			if set(self.loose_input_sockets) != {
				sym.name for sym in params.symbols if sym.name in info.dim_names
			}:
				self.loose_input_sockets = {
					sym_name: sockets.ExprSocketDef(**expr_info)
					for sym_name, expr_info in params.sym_expr_infos(info).items()
				}

		elif self.loose_input_sockets:
			self.loose_input_sockets = {}


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExportDataFile,
	DataFileExporterNode,
]
BL_NODES = {
	ct.NodeType.DataFileExporter: (ct.NodeCategory.MAXWELLSIM_OUTPUTS_FILEEXPORTERS)
}
