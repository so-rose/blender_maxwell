import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class AnyBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.Any
	socket_color = (0.0, 0.0, 0.0, 1.0)
	
	bl_label = "Any"
	
	compatible_types = {
		typ.Any: {},
	}
	
	####################
	# - Socket UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text: str) -> None:
		"""Draw the value of the real number.
		"""
		label_col_row.label(text=text)
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> None:
		return None
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		pass

####################
# - Socket Configuration
####################
class AnySocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.Any
	label: str
	
	def init(self, bl_socket: AnyBLSocket) -> None:
		pass

####################
# - Blender Registration
####################
BL_REGISTER = [
	AnyBLSocket,
]
