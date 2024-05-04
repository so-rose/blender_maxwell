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

import typing as typ

import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events

log = logger.get(__name__)


class SphereStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.SphereStructure
	bl_label = 'Sphere Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.ExprSocketDef(
			shape=(3,),
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([0, 0, 0]),
		),
		'Radius': sockets.ExprSocketDef(
			physical_type=spux.PhysicalType.Length,
			default_unit=spu.nanometer,
			default_value=150,
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Center', 'Radius', 'Medium'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Radius': 'Tidy3DUnits',
		},
	)
	def compute_structure(self, input_sockets, unit_systems) -> td.Box:
		return td.Structure(
			geometry=td.Sphere(
				radius=input_sockets['Radius'],
				center=input_sockets['Center'],
			),
			medium=input_sockets['Medium'],
		)

	####################
	# - Preview
	####################
	@events.on_value_changed(
		prop_name='preview_active',
		run_on_init=True,
		props={'preview_active'},
		managed_objs={'mesh'},
	)
	def on_preview_changed(self, props, managed_objs) -> None:
		mesh = managed_objs['mesh']

		# Push Preview State to Managed Mesh
		if props['preview_active']:
			mesh.show_preview()
		else:
			mesh.hide_preview()

	@events.on_value_changed(
		socket_name={'Center', 'Radius'},
		run_on_init=True,
		input_sockets={'Center', 'Radius'},
		managed_objs={'mesh', 'modifier'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_inputs_changed(
		self,
		managed_objs,
		input_sockets,
		unit_systems,
	):
		mesh = managed_objs['mesh']
		modifier = managed_objs['modifier']
		center = input_sockets['Center']
		radius = input_sockets['Radius']
		unit_system = unit_systems['BlenderUnits']

		# Push Loose Input Values to GeoNodes Modifier
		modifier.bl_modifier(
			mesh.bl_object(location=center),
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.StructurePrimitiveSphere),
				'inputs': {'Radius': radius},
				'unit_system': unit_system,
			},
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	SphereStructureNode,
]
BL_NODES = {
	ct.NodeType.SphereStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
