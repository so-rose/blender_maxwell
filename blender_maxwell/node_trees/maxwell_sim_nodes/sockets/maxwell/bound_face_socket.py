import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellBoundFaceBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellBoundFace
	bl_label = "Maxwell Bound Face"
	
	####################
	# - Properties
	####################
	default_choice: bpy.props.EnumProperty(
		name="Bound Face",
		description="A choice of default boundary face",
		items=[
			("PML", "PML", "Perfectly matched layer"),
			("PEC", "PEC", "Perfect electrical conductor"),
			("PMC", "PMC", "Perfect magnetic conductor"),
			("PERIODIC", "Periodic", "Infinitely periodic layer"),
		],
		default="PML",
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, "default_choice", text="")
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> td.BoundarySpec:
		return {
			"PML": td.PML(num_layers=12),
			"PEC": td.PECBoundary(),
			"PMC": td.PMCBoundary(),
			"PERIODIC": td.Periodic(),
		}[self.default_choice]
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		return None

####################
# - Socket Configuration
####################
class MaxwellBoundFaceSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellBoundFace
	label: str
	
	default_choice: str = "PML"
	
	def init(self, bl_socket: MaxwellBoundFaceBLSocket) -> None:
		bl_socket.default_choice = self.default_choice

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundFaceBLSocket,
]
