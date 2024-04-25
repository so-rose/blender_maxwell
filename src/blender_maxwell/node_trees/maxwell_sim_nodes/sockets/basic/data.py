import enum
import typing as typ

import bpy

from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from .. import base

log = logger.get(__name__)


class DataInfoColumn(enum.StrEnum):
	Length = enum.auto()
	MathType = enum.auto()
	Unit = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		return {
			DataInfoColumn.Length: 'L',
			DataInfoColumn.MathType: '∈',
			DataInfoColumn.Unit: 'U',
		}[value]

	@staticmethod
	def to_icon(value: typ.Self) -> str:
		return {
			DataInfoColumn.Length: '',
			DataInfoColumn.MathType: '',
			DataInfoColumn.Unit: '',
		}[value]


####################
# - Blender Socket
####################
class DataBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.Data
	bl_label = 'Data'
	use_info_draw = True

	####################
	# - Properties: Format
	####################
	format: str = bl_cache.BLField('')
	## TODO: typ.Literal['xarray', 'jax']

	show_info_columns: bool = bl_cache.BLField(
		False,
		prop_ui=True,
	)
	info_columns: DataInfoColumn = bl_cache.BLField(
		{DataInfoColumn.MathType, DataInfoColumn.Length}, prop_ui=True, enum_many=True
	)

	####################
	# - FlowKind
	####################
	@property
	def capabilities(self) -> ct.CapabilitiesFlow:
		return ct.CapabilitiesFlow(
			socket_type=self.socket_type,
			active_kind=self.active_kind,
			must_match={'format': self.format},
		)

	####################
	# - UI
	####################
	def draw_input_label_row(self, row: bpy.types.UILayout, text) -> None:
		if self.format == 'jax':
			row.label(text=text)
			row.prop(self, self.blfields['info_columns'])
			row.prop(
				self,
				self.blfields['show_info_columns'],
				toggle=True,
				text='',
				icon=ct.Icon.ToggleSocketInfo,
			)

	def draw_output_label_row(self, row: bpy.types.UILayout, text) -> None:
		if self.format == 'jax':
			row.prop(
				self,
				self.blfields['show_info_columns'],
				toggle=True,
				text='',
				icon=ct.Icon.ToggleSocketInfo,
			)
			row.prop(self, self.blfields['info_columns'])
			row.label(text=text)

	def draw_info(self, info: ct.InfoFlow, col: bpy.types.UILayout) -> None:
		if self.format == 'jax' and info.dim_names and self.show_info_columns:
			row = col.row()
			box = row.box()
			grid = box.grid_flow(
				columns=len(self.info_columns) + 1,
				row_major=True,
				even_columns=True,
				# even_rows=True,
				align=True,
			)

			# Dimensions
			for dim_name in info.dim_names:
				dim_idx = info.dim_idx[dim_name]
				grid.label(text=dim_name)
				if DataInfoColumn.Length in self.info_columns:
					grid.label(text=str(len(dim_idx)))
				if DataInfoColumn.MathType in self.info_columns:
					grid.label(text=spux.MathType.to_str(dim_idx.mathtype))
				if DataInfoColumn.Unit in self.info_columns:
					grid.label(text=spux.sp_to_str(dim_idx.unit))

			# Outputs
			for output_name in info.output_names:
				grid.label(text=output_name)
				if DataInfoColumn.Length in self.info_columns:
					grid.label(text='', icon=ct.Icon.DataSocketOutput)
				if DataInfoColumn.MathType in self.info_columns:
					grid.label(
						text=spux.MathType.to_str(info.output_mathtypes[output_name])
					)
				if DataInfoColumn.Unit in self.info_columns:
					grid.label(text=spux.sp_to_str(info.output_units[output_name]))


####################
# - Socket Configuration
####################
class DataSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.Data

	format: typ.Literal['xarray', 'jax', 'monitor_data']

	def init(self, bl_socket: DataBLSocket) -> None:
		bl_socket.format = self.format


####################
# - Blender Registration
####################
BL_REGISTER = [
	DataBLSocket,
]
