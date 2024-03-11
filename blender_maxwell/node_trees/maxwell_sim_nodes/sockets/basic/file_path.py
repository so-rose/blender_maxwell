import typing as typ
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts as ct

####################
# - Blender Socket
####################
class FilePathBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.FilePath
	bl_label = "File Path"
	
	####################
	# - Properties
	####################
	raw_value: bpy.props.StringProperty(
		name="File Path",
		description="Represents the path to a file",
		subtype="FILE_PATH",
		update=(lambda self, context: self.sync_prop("raw_value", context)),
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
	def value(self) -> Path:
		return Path(bpy.path.abspath(self.raw_value))
	
	@value.setter
	def value(self, value: Path) -> None:
		self.raw_value = bpy.path.relpath(str(value))

####################
# - Socket Configuration
####################
class FilePathSocketDef(pyd.BaseModel):
	socket_type: ct.SocketType = ct.SocketType.FilePath
	
	default_path: Path = Path("")
	
	def init(self, bl_socket: FilePathBLSocket) -> None:
		bl_socket.value = self.default_path

####################
# - Blender Registration
####################
BL_REGISTER = [
	FilePathBLSocket,
]
