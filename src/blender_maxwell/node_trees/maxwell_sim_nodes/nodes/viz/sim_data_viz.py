import typing as typ

import tidy3d as td
import numpy as np
import sympy as sp
import sympy.physics.units as spu

import bpy

from .....utils import analyze_geonodes
from ... import bl_socket_map
from ... import contracts as ct
from ... import sockets
from .. import base
from ... import managed_objs

CACHE = {}


class FDTDSimDataVizNode(base.MaxwellSimNode):
	node_type = ct.NodeType.FDTDSimDataViz
	bl_label = 'FDTD Sim Data Viz'

	####################
	# - Sockets
	####################
	input_sockets = {
		'FDTD Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
	}
	output_sockets = {'Preview': sockets.AnySocketDef()}

	managed_obj_defs = {
		'viz_plot': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLImage(name),
			name_prefix='',
		),
		'viz_object': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix='',
		),
	}

	####################
	# - Properties
	####################
	viz_monitor_name: bpy.props.EnumProperty(
		name='Viz Monitor Name',
		description='Monitor to visualize within the attached SimData',
		items=lambda self, context: self.retrieve_monitors(context),
		update=(lambda self, context: self.sync_viz_monitor_name(context)),
	)
	cache_viz_monitor_type: bpy.props.StringProperty(
		name='Viz Monitor Type',
		description='Type of the viz monitor',
		default='',
	)

	# Field Monitor Type
	field_viz_component: bpy.props.EnumProperty(
		name='Field Component',
		description='Field component to visualize',
		items=[
			('E', 'E', 'Electric'),
			# ("H", "H", "Magnetic"),
			# ("S", "S", "Poynting"),
			('Ex', 'Ex', 'Ex'),
			('Ey', 'Ey', 'Ey'),
			('Ez', 'Ez', 'Ez'),
			# ("Hx", "Hx", "Hx"),
			# ("Hy", "Hy", "Hy"),
			# ("Hz", "Hz", "Hz"),
		],
		default='E',
		update=lambda self, context: self.sync_prop(
			'field_viz_component', context
		),
	)
	field_viz_part: bpy.props.EnumProperty(
		name='Field Part',
		description='Field part to visualize',
		items=[
			('real', 'Real', 'Electric'),
			('imag', 'Imaginary', 'Imaginary'),
			('abs', 'Abs', 'Abs'),
			('abs^2', 'Squared Abs', 'Square Abs'),
			('phase', 'Phase', 'Phase'),
		],
		default='real',
		update=lambda self, context: self.sync_prop('field_viz_part', context),
	)
	field_viz_scale: bpy.props.EnumProperty(
		name='Field Scale',
		description='Field scale to visualize in, Linear or Log',
		items=[
			('lin', 'Linear', 'Linear Scale'),
			('dB', 'Log (dB)', 'Logarithmic (dB) Scale'),
		],
		default='lin',
		update=lambda self, context: self.sync_prop(
			'field_viz_scale', context
		),
	)
	field_viz_structure_visibility: bpy.props.FloatProperty(
		name='Field Viz Plot: Structure Visibility',
		description='Visibility of structes',
		default=0.2,
		min=0.0,
		max=1.0,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fixed_f', context
		),
	)

	field_viz_plot_fix_x: bpy.props.BoolProperty(
		name='Field Viz Plot: Fix X',
		description='Fix the x-coordinate on the plot',
		default=False,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fix_x', context
		),
	)
	field_viz_plot_fix_y: bpy.props.BoolProperty(
		name='Field Viz Plot: Fix Y',
		description='Fix the y coordinate on the plot',
		default=False,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fix_y', context
		),
	)
	field_viz_plot_fix_z: bpy.props.BoolProperty(
		name='Field Viz Plot: Fix Z',
		description='Fix the z coordinate on the plot',
		default=False,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fix_z', context
		),
	)
	field_viz_plot_fix_f: bpy.props.BoolProperty(
		name='Field Viz Plot: Fix Freq',
		description='Fix the frequency coordinate on the plot',
		default=False,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fix_f', context
		),
	)

	field_viz_plot_fixed_x: bpy.props.FloatProperty(
		name='Field Viz Plot: Fix X',
		description='Fix the x-coordinate on the plot',
		default=0.0,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fixed_x', context
		),
	)
	field_viz_plot_fixed_y: bpy.props.FloatProperty(
		name='Field Viz Plot: Fixed Y',
		description='Fix the y coordinate on the plot',
		default=0.0,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fixed_y', context
		),
	)
	field_viz_plot_fixed_z: bpy.props.FloatProperty(
		name='Field Viz Plot: Fixed Z',
		description='Fix the z coordinate on the plot',
		default=0.0,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fixed_z', context
		),
	)
	field_viz_plot_fixed_f: bpy.props.FloatProperty(
		name='Field Viz Plot: Fixed Freq (Thz)',
		description='Fix the frequency coordinate on the plot',
		default=0.0,
		update=lambda self, context: self.sync_prop(
			'field_viz_plot_fixed_f', context
		),
	)

	####################
	# - Derived Properties
	####################
	def sync_viz_monitor_name(self, context):
		if (sim_data := self._compute_input('FDTD Sim Data')) is None:
			return

		self.cache_viz_monitor_type = sim_data.monitor_data[
			self.viz_monitor_name
		].type
		self.sync_prop('viz_monitor_name', context)

	def retrieve_monitors(self, context) -> list[tuple]:
		global CACHE
		if not CACHE.get(self.instance_id):
			sim_data = self._compute_input('FDTD Sim Data')

			if sim_data is not None:
				CACHE[self.instance_id] = {
					'monitors': list(sim_data.monitor_data.keys())
				}
			else:
				return [('NONE', 'None', 'No monitors')]

		monitor_names = CACHE[self.instance_id]['monitors']

		# Check for No Monitors
		if not monitor_names:
			return [('NONE', 'None', 'No monitors')]

		return [
			(
				monitor_name,
				monitor_name,
				f"Monitor '{monitor_name}' recorded by the FDTD Sim",
			)
			for monitor_name in monitor_names
		]

	####################
	# - UI
	####################
	def draw_props(self, context, layout):
		row = layout.row()
		row.prop(self, 'viz_monitor_name', text='')
		if self.cache_viz_monitor_type == 'FieldData':
			# Array Selection
			split = layout.split(factor=0.45)
			col = split.column(align=False)
			col.label(text='Component')
			col.label(text='Part')
			col.label(text='Scale')

			col = split.column(align=False)
			col.prop(self, 'field_viz_component', text='')
			col.prop(self, 'field_viz_part', text='')
			col.prop(self, 'field_viz_scale', text='')

			# Coordinate Fixing
			split = layout.split(factor=0.45)
			col = split.column(align=False)
			col.prop(self, 'field_viz_plot_fix_x', text='Fix x (um)')
			col.prop(self, 'field_viz_plot_fix_y', text='Fix y (um)')
			col.prop(self, 'field_viz_plot_fix_z', text='Fix z (um)')
			col.prop(self, 'field_viz_plot_fix_f', text='Fix f (THz)')

			col = split.column(align=False)
			col.prop(self, 'field_viz_plot_fixed_x', text='')
			col.prop(self, 'field_viz_plot_fixed_y', text='')
			col.prop(self, 'field_viz_plot_fixed_z', text='')
			col.prop(self, 'field_viz_plot_fixed_f', text='')

	####################
	# - On Value Changed Methods
	####################
	@base.on_value_changed(
		socket_name='FDTD Sim Data',
		managed_objs={'viz_object'},
		input_sockets={'FDTD Sim Data'},
	)
	def on_value_changed__fdtd_sim_data(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
		input_sockets: dict[str, typ.Any],
	) -> None:
		global CACHE

		if (sim_data := input_sockets['FDTD Sim Data']) is None:
			CACHE.pop(self.instance_id, None)
			return

		CACHE[self.instance_id] = {
			'monitors': list(sim_data.monitor_data.keys())
		}

	####################
	# - Plotting
	####################
	@base.on_show_plot(
		managed_objs={'viz_plot'},
		props={
			'viz_monitor_name',
			'field_viz_component',
			'field_viz_part',
			'field_viz_scale',
			'field_viz_structure_visibility',
			'field_viz_plot_fix_x',
			'field_viz_plot_fix_y',
			'field_viz_plot_fix_z',
			'field_viz_plot_fix_f',
			'field_viz_plot_fixed_x',
			'field_viz_plot_fixed_y',
			'field_viz_plot_fixed_z',
			'field_viz_plot_fixed_f',
		},
		input_sockets={'FDTD Sim Data'},
		stop_propagation=True,
	)
	def on_show_plot(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
		input_sockets: dict[str, typ.Any],
		props: dict[str, typ.Any],
	):
		if (sim_data := input_sockets['FDTD Sim Data']) is None or (
			monitor_name := props['viz_monitor_name']
		) == 'NONE':
			return

		coord_fix = {}
		for coord in ['x', 'y', 'z', 'f']:
			if props[f'field_viz_plot_fix_{coord}']:
				coord_fix |= {
					coord: props[f'field_viz_plot_fixed_{coord}'],
				}

		if 'f' in coord_fix:
			coord_fix['f'] *= 1e12

		managed_objs['viz_plot'].mpl_plot_to_image(
			lambda ax: sim_data.plot_field(
				monitor_name,
				props['field_viz_component'],
				val=props['field_viz_part'],
				scale=props['field_viz_scale'],
				eps_alpha=props['field_viz_structure_visibility'],
				phase=0,
				**coord_fix,
				ax=ax,
			),
			bl_select=True,
		)

	# @base.on_show_preview(
	# managed_objs={"viz_object"},
	# )
	# def on_show_preview(
	# self,
	# managed_objs: dict[str, ct.schemas.ManagedObj],
	# ):
	# """Called whenever a Loose Input Socket is altered.
	#
	# Synchronizes the change to the actual GeoNodes modifier, so that the change is immediately visible.
	# """
	# managed_objs["viz_object"].show_preview("MESH")


####################
# - Blender Registration
####################
BL_REGISTER = [
	FDTDSimDataVizNode,
]
BL_NODES = {ct.NodeType.FDTDSimDataViz: (ct.NodeCategory.MAXWELLSIM_VIZ)}
