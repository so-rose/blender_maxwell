import typing as typ

import bpy
import pydantic as pyd
import tidy3d as td

from .. import base
from ... import contracts

class MaxwellMediumBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.MaxwellMedium
	bl_label = "Maxwell Medium"
	
	compatible_types = {
		td.components.medium.AbstractMedium: {}
	}
	
	####################
	# - Properties
	####################
	rel_permittivity: bpy.props.FloatProperty(
		name="Permittivity",
		description="Represents a simple, real permittivity.",
		default=0.0,
		precision=4,
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		"""Draw the value of the area, including a toggle for
		specifying the active unit.
		"""
		col_row = col.row(align=True)
		col_row.prop(self, "rel_permittivity", text="ϵr")
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> td.Medium:
		"""Return the built-in medium representation as a `tidy3d` object,
		ready to use in the simulation.
		
		Returns:
			A completely normal medium with permittivity set.
		"""
		
		return td.Medium(
			permittivity=self.rel_permittivity,
		)
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		"""Set the built-in medium representation by adjusting the
		permittivity, ONLY.
		
		Args:
			value: Must be a tidy3d.Medium, or similar subclass.
		"""
		
		# ONLY Allow td.Medium
		if isinstance(value, td.Medium):
			self.rel_permittivity = value.permittivity
		
		msg = f"Tried setting MaxwellMedium socket ({self}) to something that isn't a simple `tidy3d.Medium`"
		raise ValueError(msg)

####################
# - Socket Configuration
####################
class MaxwellMediumSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.MaxwellMedium
	label: str
	
	rel_permittivity: float = 1.0
	
	def init(self, bl_socket: MaxwellMediumBLSocket) -> None:
		bl_socket.rel_permittivity = self.rel_permittivity

####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMediumBLSocket,
]
