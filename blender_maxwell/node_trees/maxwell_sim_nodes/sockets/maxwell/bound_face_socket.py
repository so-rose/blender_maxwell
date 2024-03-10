import typing as typ
import typing_extensions as typx

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts as ct

class MaxwellBoundFaceBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellBoundFace
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
		update=(lambda self, context: self.sync_prop("default_choice", context)),
	)
	
	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, "default_choice", text="")
	
	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> td.BoundarySpec:
		return {
			"PML": td.PML(num_layers=12),
			"PEC": td.PECBoundary(),
			"PMC": td.PMCBoundary(),
			"PERIODIC": td.Periodic(),
		}[self.default_choice]
	
	@value.setter
	def value(self, value: typx.Literal["PML", "PEC", "PMC", "PERIODIC"]) -> None:
		self.default_choice = value

####################
# - Socket Configuration
####################
class MaxwellBoundFaceSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.MaxwellBoundFace
	
	default_choice: typx.Literal["PML", "PEC", "PMC", "PERIODIC"] = "PML"
	
	def init(self, bl_socket: MaxwellBoundFaceBLSocket) -> None:
		bl_socket.value = self.default_choice

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundFaceBLSocket,
]
