import typing as typ

import bpy
import pydantic as pyd

from .. import base
from ... import contracts as ct


class MaxwellStructureBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellStructure
	bl_label = 'Maxwell Structure'


####################
# - Socket Configuration
####################
class MaxwellStructureSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellStructure

	is_list: bool = False

	def init(self, bl_socket: MaxwellStructureBLSocket) -> None:
		bl_socket.is_list = self.is_list


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellStructureBLSocket,
]
