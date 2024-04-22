import typing as typ

import bpy

from blender_maxwell.utils import bl_cache
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from .. import base


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

	@property
	def value(self):
		return None

	####################
	# - UI
	####################
	def draw_info(self, info: ct.InfoFlow, col: bpy.types.UILayout) -> None:
		if self.format == 'jax' and info.dim_names:
			row = col.row()
			box = row.box()
			grid = box.grid_flow(
				columns=3,
				row_major=True,
				even_columns=True,
				#even_rows=True,
				align=True,
			)

			# Grid Header
			#grid.label(text='Dim')
			#grid.label(text='Len')
			#grid.label(text='Unit')

			# Dimension Names
			for dim_name in info.dim_names:
				dim_idx = info.dim_idx[dim_name]
				grid.label(text=dim_name)
				grid.label(text=str(len(dim_idx)))
				grid.label(text=spux.sp_to_str(dim_idx.unit))


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
