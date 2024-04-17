import typing as typ

import bpy
import jax.numpy as jnp

from .....utils import logger
from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class VizNode(base.MaxwellSimNode):
	"""Node for visualizing simulation data, by querying its monitors."""

	node_type = ct.NodeType.Viz
	bl_label = 'Viz'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Data': sockets.AnySocketDef(),
		'Freq': sockets.PhysicalFreqSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Preview': sockets.AnySocketDef(),
	}

	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	#####################
	## - Properties
	#####################
	colormap: bpy.props.EnumProperty(
		name='Colormap',
		description='Colormap to apply to grayscale output',
		items=[
			('VIRIDIS', 'Viridis', 'Good default colormap'),
			('GRAYSCALE', 'Grayscale', 'Barebones'),
		],
		default='VIRIDIS',
		update=lambda self, context: self.sync_prop('colormap', context),
	)

	#####################
	## - UI
	#####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout):
		col.prop(self, 'colormap')

	#####################
	## - Plotting
	#####################
	@events.on_show_plot(
		managed_objs={'plot'},
		input_sockets={'Data', 'Freq'},
		props={'colormap'},
		unit_systems={'Tidy3DUnits': ct.UNITS_TIDY3D},
		scale_input_sockets={
			'Freq': 'Tidy3DUnits',
		},
		stop_propagation=True,
	)
	def on_show_plot(
		self,
		managed_objs: dict,
		input_sockets: dict,
		props: dict,
		unit_systems: dict,
	):
		managed_objs['plot'].map_2d_to_image(
			input_sockets['Data'].as_bound_jax_func(),
			colormap=props['colormap'],
			bl_select=True,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	VizNode,
]
BL_NODES = {ct.NodeType.Viz: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
