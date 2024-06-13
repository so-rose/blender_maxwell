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

import enum
import functools
import typing as typ
from pathlib import Path

import bpy
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events

log = logger.get(__name__)

FK = ct.FlowKind
FS = ct.FlowSignal


class ValidTDFileExts(enum.StrEnum):
	"""Valid importable Tidy3D file extensions."""

	SimData = enum.auto()
	Sim = enum.auto()
	Medium = enum.auto()

	@staticmethod
	def to_name(value: typ.Self) -> str:
		VFE = ValidTDFileExts
		return {
			VFE.SimData: 'Sim Data',
			VFE.Sim: 'Sim',
			VFE.Medium: 'Medium',
		}[value]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		return ''

	def bl_enum_element(self, i: int) -> ct.BLEnumElement:
		return (
			str(self),
			ValidTDFileExts.to_name(self),
			ValidTDFileExts.to_name(self),
			ValidTDFileExts.to_icon(self),
			i,
		)

	####################
	# - Properties
	####################
	@functools.cached_property
	def valid_exts(self) -> set[str]:
		VFE = ValidTDFileExts
		match self:
			case VFE.SimData:
				return {
					'.hdf5.gz',
					'.hdf5',
				}

			case VFE.Sim | VFE.Medium:
				return {
					'.hdf5.gz',
					'.hdf5',
					'.json',
					'.yaml',
				}

		raise TypeError

	@functools.cached_property
	def td_type(self) -> set[str]:
		"""The corresponding Tidy3D type."""
		VFE = ValidTDFileExts
		return {
			VFE.SimData: td.SimulationData,
			VFE.Sim: td.Simulation,
			VFE.Medium: td.Medium,
		}[self]

	####################
	# - Methods
	####################
	def is_path_compatible(self, path: Path) -> bool:
		ext_matches = ''.join(path.suffixes) in self.valid_exts
		return ext_matches and path.is_file()

	def load(self, file_path: Path) -> typ.Any:
		VFE = ValidTDFileExts
		match self:
			case VFE.SimData | VFE.Sim | VFE.Medium:
				return self.td_type.from_file(str(file_path))

		raise TypeError


####################
# - Node
####################
class Tidy3DFileImporterNode(base.MaxwellSimNode):
	"""Import a simulation design or analysis element from a Tidy3D object."""

	node_type = ct.NodeType.Tidy3DFileImporter
	bl_label = 'Tidy3D File Importer'

	input_sockets: typ.ClassVar = {
		'File Path': sockets.FilePathSocketDef(),
	}
	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	####################
	# - Properties
	####################
	tidy3d_type: ValidTDFileExts = bl_cache.BLField(ValidTDFileExts.SimData)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout):
		col.prop(self, 'tidy3d_type', text='')

	####################
	# - Event Methods: Setup Output Socket
	####################
	@events.on_value_changed(
		prop_name='tidy3d_type',
		run_on_init=True,
		# Loaded
		props={'tidy3d_type'},
	)
	def on_file_changed(self, props) -> None:
		self.loose_output_sockets = {
			ValidTDFileExts.SimData: {
				'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
			},
			ValidTDFileExts.Sim: {'Sim': sockets.MaxwellFDTDSimSocketDef()},
			ValidTDFileExts.Medium: {'Medium': sockets.MaxwellMediumSocketDef()},
		}[props['tidy3d_type']]

	####################
	# - FlowKind.Value
	####################
	def _compute_td_obj(
		self, props, input_sockets
	) -> td.components.base.Tidy3dBaseModel:
		tidy3d_type = props['tidy3d_type']
		file_path = input_sockets['File Path']

		if tidy3d_type.is_path_compatible(file_path):
			return tidy3d_type.load(file_path)
		return FS.FlowPending

	@events.computes_output_socket(
		'Sim',
		kind=FK.Value,
		# Loaded
		props={'tidy3d_type'},
		inscks_kinds={'File Path': FK.Value},
	)
	def compute_sim(self, props, input_sockets) -> td.Simulation:
		return self._compute_td_obj(props, input_sockets)

	@events.computes_output_socket(
		'Sim Data',
		kind=FK.Value,
		# Loaded
		props={'tidy3d_type'},
		inscks_kinds={'File Path': FK.Value},
	)
	def compute_sim_data(self, props, input_sockets) -> td.SimulationData:
		return self._compute_td_obj(props, input_sockets)

	@events.computes_output_socket(
		'Medium',
		kind=FK.Value,
		# Loaded
		props={'tidy3d_type'},
		inscks_kinds={'File Path': FK.Value},
	)
	def compute_medium(self, props, input_sockets: dict) -> td.Medium:
		return self._compute_td_obj(props, input_sockets)


####################
# - Blender Registration
####################
BL_REGISTER = [
	Tidy3DFileImporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DFileImporter: (ct.NodeCategory.MAXWELLSIM_INPUTS_FILEIMPORTERS)
}
