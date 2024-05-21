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

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.assets.geonodes import GeoNodes, import_geonodes
from blender_maxwell.utils import bl_cache, logger
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class EHFieldMonitorNode(base.MaxwellSimNode):
	"""Node providing for the monitoring of electromagnetic fields within a given planar region or volume."""

	node_type = ct.NodeType.EHFieldMonitor
	bl_label = 'EH Field Monitor'
	use_sim_node_name = True

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Center': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=spux.PhysicalType.Length,
		),
		'Size': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			physical_type=spux.PhysicalType.Length,
			default_value=sp.Matrix([1, 1, 1]),
			abs_min=0,
		),
		'Stride': sockets.ExprSocketDef(
			size=spux.NumberSize1D.Vec3,
			mathtype=spux.MathType.Integer,
			default_value=sp.Matrix([10, 10, 10]),
			abs_min=0,
		),
	}
	input_socket_sets: typ.ClassVar = {
		'Freq Domain': {
			'Freqs': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.Range,
				physical_type=spux.PhysicalType.Freq,
				default_unit=spux.THz,
				default_min=374.7406,  ## 800nm
				default_max=1498.962,  ## 200nm
				default_steps=100,
			),
		},
		'Time Domain': {
			't Range': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.Range,
				physical_type=spux.PhysicalType.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=0,
			),
			't Stride': sockets.ExprSocketDef(
				mathtype=spux.MathType.Integer,
				default_value=100,
			),
		},
	}
	output_socket_sets: typ.ClassVar = {
		'Freq Domain': {'Freq Monitor': sockets.MaxwellMonitorSocketDef()},
		'Time Domain': {'Time Monitor': sockets.MaxwellMonitorSocketDef()},
	}

	managed_obj_types: typ.ClassVar = {
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	fields: set[ct.SimFieldPols] = bl_cache.BLField(set(ct.SimFieldPols))

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['fields'], expand=True)

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'Freq Monitor',
		props={'sim_node_name', 'fields'},
		input_sockets={
			'Center',
			'Size',
			'Stride',
			'Freqs',
		},
		input_socket_kinds={
			'Freqs': ct.FlowKind.Range,
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
			'Freqs': 'Tidy3DUnits',
		},
	)
	def compute_freq_monitor(
		self,
		input_sockets: dict,
		props: dict,
		unit_systems: dict,
	) -> td.FieldMonitor:
		log.info(
			'Computing FieldMonitor (name="%s") with center="%s", size="%s"',
			props['sim_node_name'],
			input_sockets['Center'],
			input_sockets['Size'],
		)
		return td.FieldMonitor(
			center=input_sockets['Center'],
			size=input_sockets['Size'],
			name=props['sim_node_name'],
			interval_space=tuple(input_sockets['Stride']),
			freqs=input_sockets['Freqs'].realize().values,
			fields=props['fields'],
		)

	@events.computes_output_socket(
		'Time Monitor',
		props={'sim_node_name', 'fields'},
		input_sockets={
			'Center',
			'Size',
			'Stride',
			't Range',
			't Stride',
		},
		input_socket_kinds={
			't Range': ct.FlowKind.Range,
		},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Center': 'Tidy3DUnits',
			'Size': 'Tidy3DUnits',
			't Range': 'Tidy3DUnits',
		},
	)
	def compute_time_monitor(
		self,
		input_sockets: dict,
		props: dict,
		unit_systems: dict,
	) -> td.FieldMonitor:
		log.info(
			'Computing FieldMonitor (name="%s") with center="%s", size="%s"',
			props['sim_node_name'],
			input_sockets['Center'],
			input_sockets['Size'],
		)
		return td.FieldTimeMonitor(
			center=input_sockets['Center'],
			size=input_sockets['Size'],
			name=props['sim_node_name'],
			interval_space=tuple(input_sockets['Stride']),
			start=input_sockets['t Range'].realize_start(),
			stop=input_sockets['t Range'].realize_stop(),
			interval=input_sockets['t Stride'],
			fields=props['fields'],
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
		managed_objs={'modifier'},
		input_sockets={'Center', 'Size'},
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
		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.MonitorEHField),
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
	EHFieldMonitorNode,
]
BL_NODES = {ct.NodeType.EHFieldMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
