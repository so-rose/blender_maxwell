import typing as typ

import bpy
import sympy.physics.units as spu
import pydantic as pyd
import tidy3d as td
import scipy as sc

from .. import base
from ... import contracts as ct

VAC_SPEED_OF_LIGHT = (
	sc.constants.speed_of_light
	* spu.meter/spu.second
)

class MaxwellMonitorBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMonitor
	bl_label = "Maxwell Monitor"
	use_units = True
	
	####################
	# - Properties
	####################
	wl: bpy.props.FloatProperty(
		name="WL",
		description="WL to store in monitor",
		default=500.0,
		precision=4,
		step=50,
		update=(lambda self, context: self.sync_prop("wl", context)),
	)
	
	@property
	def value(self) -> td.Monitor:
		freq = spu.convert_to(
			VAC_SPEED_OF_LIGHT / (self.wl*self.unit),
			spu.hertz,
		) / spu.hertz
		return td.FieldMonitor(
			size=(td.inf, td.inf, 0),
			freqs=[freq],
			name="fields",
			colocate=True,
		)

####################
# - Socket Configuration
####################
class MaxwellMonitorSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMonitor
	
	def init(self, bl_socket: MaxwellMonitorBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMonitorBLSocket,
]
