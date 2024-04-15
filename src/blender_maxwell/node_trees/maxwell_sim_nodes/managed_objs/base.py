import abc
import typing as typ

from ....utils import serialize
from .. import contracts as ct


class ManagedObj(abc.ABC):
	managed_obj_type: ct.ManagedObjType

	@abc.abstractmethod
	def __init__(
		self,
		name: ct.ManagedObjName,
	):
		"""Initializes the managed object with a unique name."""

	####################
	# - Properties
	####################
	@property
	@abc.abstractmethod
	def name(self) -> str:
		"""Retrieve the name of the managed object."""

	@name.setter
	@abc.abstractmethod
	def name(self, value: str) -> None:
		"""Retrieve the name of the managed object."""

	####################
	# - Methods
	####################
	@abc.abstractmethod
	def free(self) -> None:
		"""Cleanup the resources managed by the managed object."""

	@abc.abstractmethod
	def bl_select(self) -> None:
		"""Select the managed object in Blender, if such an operation makes sense."""

	@abc.abstractmethod
	def hide_preview(self) -> None:
		"""Select the managed object in Blender, if such an operation makes sense."""

	####################
	# - Serialization
	####################
	def dump_as_msgspec(self) -> serialize.NaiveRepresentation:
		return [
			serialize.TypeID.ManagedObj,
			self.__class__.__name__,
			(self.managed_obj_type, self.name),
		]

	@staticmethod
	def parse_as_msgspec(obj: serialize.NaiveRepresentation) -> typ.Self:
		return ManagedObj.__subclasses__[obj[1]](**obj[2])
