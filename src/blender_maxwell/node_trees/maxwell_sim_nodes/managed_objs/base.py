# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import abc
import typing as typ

from blender_maxwell.utils import logger, serialize

from .. import contracts as ct

log = logger.get(__name__)


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
			self.name,
		]

	@staticmethod
	def parse_as_msgspec(obj: serialize.NaiveRepresentation) -> typ.Self:
		return next(
			subclass(obj[2])
			for subclass in ManagedObj.__subclasses__()
			if subclass.__name__ == obj[1]
		)
