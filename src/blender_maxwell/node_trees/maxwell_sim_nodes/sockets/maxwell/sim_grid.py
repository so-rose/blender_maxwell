import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts as ct


class MaxwellSimGridBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellSimGrid
	bl_label = 'Maxwell Sim Grid'

	####################
	# - Properties
	####################
	min_steps_per_wl: bpy.props.FloatProperty(
		name='Minimum Steps per Wavelength',
		description='How many grid steps to ensure per wavelength',
		default=10.0,
		min=0.01,
		# step=10,
		precision=2,
		update=(
			lambda self, context: self.sync_prop('min_steps_per_wl', context)
		),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		split = col.split(factor=0.5, align=False)

		col = split.column(align=True)
		col.label(text='min. stp/λ')

		col = split.column(align=True)
		col.prop(self, 'min_steps_per_wl', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> td.GridSpec:
		return td.GridSpec.auto(
			min_steps_per_wvl=self.min_steps_per_wl,
		)


####################
# - Socket Configuration
####################
class MaxwellSimGridSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellSimGrid

	min_steps_per_wl: float = 10.0

	def init(self, bl_socket: MaxwellSimGridBLSocket) -> None:
		bl_socket.min_steps_per_wl = self.min_steps_per_wl


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellSimGridBLSocket,
]
