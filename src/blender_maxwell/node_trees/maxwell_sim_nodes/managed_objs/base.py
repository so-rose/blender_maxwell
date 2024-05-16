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
	"""A weak name-based reference to some kind of object external to this software.

	While the object doesn't have to come from Blender's `bpy.types`, that is admittedly the driving motivation for this class: To encapsulate access to the powerful visual tools granted by Blender's 3D viewport, image editor, and UI.
	Through extensive testing, the functionality of an implicitly-cached, semi-strictly immediate-mode interface, demanding only a weakly-referenced name as persistance, has emerged (with all of the associated tradeoffs).

	While not suited to all use cases, the `ManagedObj` paradigm is perfect for many situations where a node needs to "loosely own" something external and non-trivial.
	Intriguingly, the precise definition of "loose" has grown to vary greatly between subclasses, as it ends of demonstrating itself to be a matter of taste more than determinism.

	This abstract base class serves to provide a few of the most basic of commonly-available - especially the `dump_as_msgspec`/`parse_as_msgspec` methods that allow it to be persisted using `blender_maxwell.utils.serialize`.

	Parameters:
		managed_obj_type: Enum identifier indicating which of the `ct.ManagedObjType` the instance should declare itself as.
	"""

	managed_obj_type: ct.ManagedObjType

	@abc.abstractmethod
	def __init__(self, name: ct.ManagedObjName, prev_name: str | None = None):
		"""Initializes the managed object with a unique name.

		Use `prev_name` to indicate that the managed object will initially be avaiable under `prev_name`, but that it should be renamed to `name`.
		"""

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
		"""Hide any active preview of the managed object, if it exists, and if such an operation makes sense."""

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
