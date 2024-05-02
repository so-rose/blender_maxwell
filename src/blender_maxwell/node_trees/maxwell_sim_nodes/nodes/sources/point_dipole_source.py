import typing as typ

import bpy
import sympy as sp
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class PointDipoleSourceNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PointDipoleSource
	bl_label = 'Point Dipole Source'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
		'Center': sockets.ExprSocketDef(
			shape=(3,),
			mathtype=spux.MathType.Real,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([0, 0, 0]),
		),
		'Interpolate': sockets.BoolSocketDef(
			default_value=True,
		),
	}
	output_sockets: typ.ClassVar = {
		'Source': sockets.MaxwellSourceSocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	pol_axis: ct.SimSpaceAxis = bl_cache.BLField(ct.SimSpaceAxis.X, prop_ui=True)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout):
		layout.prop(self, self.blfields['pol_axis'], expand=True)

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Source',
		input_sockets={'Temporal Shape', 'Center', 'Interpolate'},
		props={'pol_axis'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
		},
	)
	def compute_source(
		self,
		input_sockets: dict[str, typ.Any],
		props: dict[str, typ.Any],
		unit_systems: dict,
	) -> td.PointDipole:
		pol_axis = {
			ct.SimSpaceAxis.X: 'Ex',
			ct.SimSpaceAxis.Y: 'Ey',
			ct.SimSpaceAxis.Z: 'Ez',
		}[props['pol_axis']]

		return td.PointDipole(
			center=input_sockets['Center'],
			source_time=input_sockets['Temporal Shape'],
			interpolate=input_sockets['Interpolate'],
			polarization=pol_axis,
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
		"""Enables/disables previewing of the GeoNodes-driven mesh, regardless of whether a particular GeoNodes tree is chosen."""
		mesh = managed_objs['mesh']

		# Push Preview State to Managed Mesh
		if props['preview_active']:
			mesh.show_preview()
		else:
			mesh.hide_preview()

	@events.on_value_changed(
		socket_name={'Center'},
		prop_name='pol_axis',
		run_on_init=True,
		# Pass Data
		managed_objs={'mesh', 'modifier'},
		props={'pol_axis'},
		input_sockets={'Center'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={'Center': 'BlenderUnits'},
	)
	def on_inputs_changed(
		self, managed_objs, props, input_sockets, unit_systems
	) -> None:
		mesh = managed_objs['mesh']
		modifier = managed_objs['modifier']
		center = input_sockets['Center']
		unit_system = unit_systems['BlenderUnits']
		axis = {
			ct.SimSpaceAxis.X: 0,
			ct.SimSpaceAxis.Y: 1,
			ct.SimSpaceAxis.Z: 2,
		}[props['pol_axis']]

		# Push Loose Input Values to GeoNodes Modifier
		modifier.bl_modifier(
			mesh.bl_object(location=center),
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.SourcePointDipole),
				'inputs': {'Axis': axis},
				'unit_system': unit_system,
			},
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {ct.NodeType.PointDipoleSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
