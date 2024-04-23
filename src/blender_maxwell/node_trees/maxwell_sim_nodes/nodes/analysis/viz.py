import enum
import typing as typ

import bpy

from blender_maxwell.utils import bl_cache, image_ops, logger

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

log = logger.get(__name__)


class VizNode(base.MaxwellSimNode):
	"""Node for visualizing simulation data, by querying its monitors.

	Attributes:
		colormap: Colormap to apply to 0..1 output.

	"""

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
	colormap: image_ops.Colormap = bl_cache.BLField(
		image_ops.Colormap.Viridis, prop_ui=True
	)

	#####################
	## - UI
	#####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout):
		col.prop(self, self.blfields['colormap'], text='')

	#####################
	## - Plotting
	#####################
	@events.on_show_plot(
		managed_objs={'plot'},
		props={'colormap'},
		input_sockets={'Data'},
		input_socket_kinds={'Data': ct.FlowKind.Array},
		input_sockets_optional={'Data': True},
		stop_propagation=True,
	)
	def on_show_plot(
		self,
		managed_objs: dict,
		input_sockets: dict,
		props: dict,
	):
		if input_sockets['Data'] is not None:
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
