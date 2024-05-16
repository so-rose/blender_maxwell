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


class PowerFluxMonitorNode(base.MaxwellSimNode):
	"""Node providing for the monitoring of electromagnetic field flux a given planar region or volume, in either the frequency or the time domain."""

	node_type = ct.NodeType.PowerFluxMonitor
	bl_label = 'Power Flux Monitor'
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
				active_kind=ct.FlowKind.LazyArrayRange,
				physical_type=spux.PhysicalType.Freq,
				default_unit=spux.THz,
				default_min=374.7406,  ## 800nm
				default_max=1498.962,  ## 200nm
				default_steps=100,
			),
		},
		'Time Domain': {
			't Range': sockets.ExprSocketDef(
				active_kind=ct.FlowKind.LazyArrayRange,
				physical_type=spux.PhysicalType.Time,
				default_unit=spu.picosecond,
				default_min=0,
				default_max=10,
				default_steps=2,
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
		'mesh': managed_objs.ManagedBLMesh,
		'modifier': managed_objs.ManagedBLModifier,
	}

	####################
	# - Properties
	####################
	direction_2d: ct.SimAxisDir = bl_cache.BLField(ct.SimAxisDir.Plus)
	include_3d: set[ct.SimSpaceAxis] = bl_cache.BLField(set(ct.SimSpaceAxis))
	include_3d_x: set[ct.SimAxisDir] = bl_cache.BLField(set(ct.SimAxisDir))
	include_3d_y: set[ct.SimAxisDir] = bl_cache.BLField(set(ct.SimAxisDir))
	include_3d_z: set[ct.SimAxisDir] = bl_cache.BLField(set(ct.SimAxisDir))

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		# 2D Monitor
		if 0 in self._compute_input('Size'):
			layout.prop(self, self.blfields['direction_2d'], expand=True)

		# 3D Monitor
		else:
			layout.prop(self, self.blfields['include_3d'], expand=True)
			row = layout.row(align=False)
			if ct.SimSpaceAxis.X in self.include_3d:
				row.prop(self, self.blfields['include_3d_x'], expand=True)
			if ct.SimSpaceAxis.Y in self.include_3d:
				row.prop(self, self.blfields['include_3d_y'], expand=True)
			if ct.SimSpaceAxis.Z in self.include_3d:
				row.prop(self, self.blfields['include_3d_z'], expand=True)

	####################
	# - Event Methods: Computation
	####################
	@events.computes_output_socket(
		'Freq Monitor',
		props={'sim_node_name', 'direction_2d'},
		input_sockets={
			'Center',
			'Size',
			'Stride',
			'Freqs',
		},
		input_socket_kinds={
			'Freqs': ct.FlowKind.LazyArrayRange,
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
			'Computing FluxMonitor (name="%s") with center="%s", size="%s"',
			props['sim_node_name'],
			input_sockets['Center'],
			input_sockets['Size'],
		)
		return td.FluxMonitor(
			center=input_sockets['Center'],
			size=input_sockets['Size'],
			name=props['sim_node_name'],
			interval_space=(1, 1, 1),
			freqs=input_sockets['Freqs'].realize_array.values,
			normal_dir=props['direction_2d'].plus_or_minus,
		)

	####################
	# - Preview - Changes to Input Sockets
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
		prop_name={'direction_2d'},
		run_on_init=True,
		# Loaded
		managed_objs={'mesh', 'modifier'},
		props={'direction_2d'},
		input_sockets={'Center', 'Size'},
		unit_systems={'BlenderUnits': ct.UNITS_BLENDER},
		scale_input_sockets={
			'Center': 'BlenderUnits',
		},
	)
	def on_inputs_changed(self, managed_objs, props, input_sockets, unit_systems):
		# Push Input Values to GeoNodes Modifier
		managed_objs['modifier'].bl_modifier(
			'NODES',
			{
				'node_group': import_geonodes(GeoNodes.MonitorPowerFlux),
				'unit_system': unit_systems['BlenderUnits'],
				'inputs': {
					'Size': input_sockets['Size'],
					'Direction': props['direction_2d'].true_or_false,
				},
			},
			location=input_sockets['Center'],
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PowerFluxMonitorNode,
]
BL_NODES = {ct.NodeType.PowerFluxMonitor: (ct.NodeCategory.MAXWELLSIM_MONITORS)}
