import abc
import typing as typ

import bpy
import pydantic as pyd

from .....utils import serialize
from ..socket_types import SocketType


class SocketDef(pyd.BaseModel, abc.ABC):
	socket_type: SocketType

	@abc.abstractmethod
	def init(self, bl_socket: bpy.types.NodeSocket) -> None:
		"""Initializes a real Blender node socket from this socket definition."""

	####################
	# - Serialization
	####################
	def dump_as_msgspec(self) -> serialize.NaiveRepresentation:
		return [serialize.TypeID.SocketDef, self.__class__.__name__, self.model_dump()]

	@staticmethod
	def parse_as_msgspec(obj: serialize.NaiveRepresentation) -> typ.Self:
		return SocketDef.__subclasses__[obj[1]](**obj[2])
