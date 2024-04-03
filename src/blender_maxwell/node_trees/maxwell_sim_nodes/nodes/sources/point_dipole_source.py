import typing as typ

import bpy
import sympy.physics.units as spu
import tidy3d as td

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events


class PointDipoleSourceNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PointDipoleSource
	bl_label = 'Point Dipole Source'

	####################
	# - Sockets
	####################
	input_sockets = {
		'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
		'Center': sockets.PhysicalPoint3DSocketDef(),
		'Interpolate': sockets.BoolSocketDef(
			default_value=True,
		),
	}
	output_sockets = {
		'Source': sockets.MaxwellSourceSocketDef(),
	}

	managed_obj_defs = {
		'sphere_empty': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix='',
		)
	}

	####################
	# - Properties
	####################
	pol_axis: bpy.props.EnumProperty(
		name='Polarization Axis',
		description='Polarization Axis',
		items=[
			('EX', 'Ex', 'Electric field in x-dir'),
			('EY', 'Ey', 'Electric field in y-dir'),
			('EZ', 'Ez', 'Electric field in z-dir'),
		],
		default='EX',
		update=(lambda self, context: self.sync_prop('pol_axis', context)),
	)

	####################
	# - UI
	####################
	def draw_props(self, context, layout):
		split = layout.split(factor=0.6)

		col = split.column()
		col.label(text='Pol Axis')

		col = split.column()
		col.prop(self, 'pol_axis', text='')

	####################
	# - Output Socket Computation
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
		self, input_sockets: dict[str, typ.Any], props: dict[str, typ.Any]
	) -> td.PointDipole:
		pol_axis = {
			'EX': 'Ex',
			'EY': 'Ey',
			'EZ': 'Ez',
		}[props['pol_axis']]

		return td.PointDipole(
			center=input_sockets['Center'],
			source_time=input_sockets['Temporal Shape'],
			interpolate=input_sockets['Interpolate'],
			polarization=pol_axis,
		)

	#####################
	## - Preview
	#####################
	# @events.on_value_changed(
	# socket_name='Center',
	# input_sockets={'Center'},
	# managed_objs={'sphere_empty'},
	# )
	# def on_value_changed__center(
	# self,
	# input_sockets: dict,
	# managed_objs: dict[str, ct.schemas.ManagedObj],
	# ):
	# _center = input_sockets['Center']
	# center = tuple(spu.convert_to(_center, spu.um) / spu.um)
	# ## TODO: Preview unit system?? Presume um for now

	# mobj = managed_objs['sphere_empty']
	# bl_object = mobj.bl_object('EMPTY')
	# bl_object.location = center  # tuple([float(el) for el in center])

	# @events.on_show_preview(
	# managed_objs={'sphere_empty'},
	# )
	# def on_show_preview(
	# self,
	# managed_objs: dict[str, ct.schemas.ManagedObj],
	# ):
	# managed_objs['sphere_empty'].show_preview(
	# 'EMPTY',
	# empty_display_type='SPHERE',
	# )
	# managed_objs['sphere_empty'].bl_object('EMPTY').empty_display_size = 0.2


####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {ct.NodeType.PointDipoleSource: (ct.NodeCategory.MAXWELLSIM_SOURCES)}
