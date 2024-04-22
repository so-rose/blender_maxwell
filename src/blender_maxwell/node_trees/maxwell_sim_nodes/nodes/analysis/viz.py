import typing as typ

import bpy

from blender_maxwell.utils import logger

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
		'Data': sockets.DataSocketDef(format='jax'),
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
		update=lambda self, context: self.on_prop_changed('colormap', context),
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
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Array},
		props={'colormap'},
		stop_propagation=True,
	)
	def on_show_plot(
		self,
		managed_objs: dict,
		input_sockets: dict,
		props: dict,
	):
		managed_objs['plot'].map_2d_to_image(
			input_sockets['Data'].values,
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
