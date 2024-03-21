import typing as typ

from ..bl import ManagedObjName
from ..managed_obj_type import ManagedObjType


class ManagedObj(typ.Protocol):
	managed_obj_type: ManagedObjType

	def __init__(
		self,
		name: ManagedObjName,
	): ...

	@property
	def name(self) -> str: ...
	@name.setter
	def name(self, value: str): ...

	def free(self): ...

	def bl_select(self): ...
