import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from ......utils import analyze_geonodes
from .... import contracts as ct
from .....assets.import_geonodes import import_geonodes
from .... import managed_objs, sockets
from ... import base

GEONODES_BOX = 'box'


class BoxStructureNode(base.MaxwellSimNode):
	node_type = ct.NodeType.BoxStructure
	bl_label = 'Box Structure'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets = {
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Size': sockets.PhysicalSize3DSocketDef(
			default_value=sp.Matrix([500, 500, 500]) * spu.nm
		),
	}
	output_sockets: typ.ClassVar = {
		'Structure': sockets.MaxwellStructureSocketDef(),
	}

	managed_obj_defs: typ.ClassVar = {
		'mesh': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLMesh(name),
			name_prefix='',
		),
		'box': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLModifier(name),
			name_prefix='',
		),
	}

	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		'Structure',
		input_sockets={'Medium', 'Center', 'Size'},
	)
	def compute_structure(self, input_sockets: dict) -> td.Box:
		medium = input_sockets['Medium']
		center = as_unit_system(input_sockets['Center'], 'tidy3d')
		size = as_unit_system(input_sockets['Size'], 'tidy3d')
		#_center = input_sockets['Center']
		#_size = input_sockets['Size']

		#center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		#size = tuple(spu.convert_to(_size, spu.um) / spu.um)

		return td.Structure(
			geometry=td.Box(
				center=center,
				size=size,
			),
			medium=medium,
		)

	####################
	# - Events
	####################
	@base.on_value_changed(
		socket_name={'Center', 'Size'},
		input_sockets={'Center', 'Size'},
		managed_objs={'mesh', 'box'},
	)
	def on_value_changed__center_size(
		self,
		input_sockets: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		center = as_unit_system(input_sockets['Center'], 'blender')
		#center = tuple([float(el) for el in spu.convert_to(_center, spu.um) / spu.um])
		## TODO: Implement + aggressively memoize as_unit_system
		## - This should also understand that ex. Blender likes tuples, Tidy3D might like something else.

		size = as_unit_system(input_sockets['Size'], 'blender')
		#size = tuple([float(el) for el in spu.convert_to(_size, spu.um) / spu.um])

		# Sync Attributes
		managed_objs['mesh'].bl_object().location = center
		managed_objs['box'].bl_modifier(managed_objs['mesh'].bl_object(), 'NODES', {
			'node_group': import_geonodes(GEONODES_BOX, 'link'),
			'inputs': {
				'Size': size,  
			},
		})

	@base.on_show_preview(
		managed_objs={'mesh'},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		managed_objs['mesh'].show_preview()
		self.on_value_changed__center_size()


####################
# - Blender Registration
####################
BL_REGISTER = [
	BoxStructureNode,
]
BL_NODES = {
	ct.NodeType.BoxStructure: (ct.NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES)
}
