import typing as typ
import typing as typx

import pydantic as pyd

from ..bl import ManagedObjName, SocketName
from ..managed_obj_type import ManagedObjType

class ManagedObj(typ.Protocol):
	managed_obj_type: ManagedObjType
	
	def __init__(
		self,
		name: ManagedObjName,
	):
		...
	
	@property
	def name(self) -> str: ...
	@name.setter
	def name(self, value: str): ...
	
	def free(self):
		...
	
	def bl_select(self):
		"""If this is a managed Blender object, and the operation "select this in Blender" makes sense, then do so.

		Else, do nothing.
		"""
		pass
