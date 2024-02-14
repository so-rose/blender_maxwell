import typing as typ
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

####################
# - Blender Socket
####################
class FilePathBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.FilePath
	socket_color = (0.2, 0.2, 0.2, 1.0)
	
	bl_label = "File Path"
	
	compatible_types = {
		Path: {},
	}
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.StringProperty(
		name="File Path",
		description="Represents the path to a file",
		#default="",
		subtype="FILE_PATH",
	)
	
	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col_row = col.row(align=True)
		col_row.prop(self, "raw_value", text="")
	
	####################
	# - Computation of Default Value
	####################
	@property
	def default_value(self) -> Path:
		"""Return the text.
		
		Returns:
			The text as a string.
		"""
		
		return Path(str(self.raw_value))
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		"""Set the real number from some compatible type, namely
		real sympy expressions with no symbols, or floats.
		"""
		
		# (Guard) Value Compatibility
		if not self.is_compatible(value):
			msg = f"Tried setting socket ({self}) to incompatible value ({value}) of type {type(value)}"
			raise ValueError(msg)
		
		self.raw_value = str(Path(value))

####################
# - Socket Configuration
####################
class FilePathSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.FilePath
	label: str
	
	default_path: Path
	
	def init(self, bl_socket: FilePathBLSocket) -> None:
		bl_socket.default_value = self.default_path

####################
# - Blender Registration
####################
BL_REGISTER = [
	FilePathBLSocket,
]
