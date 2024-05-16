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


class BoxStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.BoxStructure
	bl_label = 'Box Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			default_unit=spu.micrometer,
			default_value=sp.Matrix([0, 0, 0]),
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			default_unit=spu.nanometer,
			default_value=sp.Matrix([500, 500, 500]),
			abs_min=0.001,
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Structure',
		input_sockets={'Medium', 'Center', 'Size'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
		},
	)
	def compute_structure(self, input_sockets, unit_systems) -> td.Box:
		return td.Structure(
			geometry=td.Box(
				center=input_sockets['Center'],
				size=input_sockets['Size'],
			),
			medium=input_sockets['Medium'],
		)

	####################
	# - Preview
	####################
	@events.on_value_changed(
		# Trigger
		prop_name='preview_active',
		# Loaded
		managed_objs={'modifier'},
		props={'preview_active'},
	)
	def on_preview_changed(self, managed_objs, props):
		if props['preview_active']:
			managed_objs['modifier'].show_preview()
		else:
			managed_objs['modifier'].hide_preview()

	@events.on_value_changed(
		# Trigger
		socket_name={'Center', 'Size'},
		run_on_init=True,
		# Loaded
		input_sockets={'Center', 'Size'},
		managed_objs={'modifier'},
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
		# Push Loose Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.StructurePrimitiveBox),
				'unit_system': unit_systems['BlenderUnits'],
				'inputs': {
					'Size': input_sockets['Size'],
				},
			},
			location=input_sockets['Center'],
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoxStructureNode,
]
BL_NODES = {
	ct.NodeType.BoxStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
