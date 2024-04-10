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
	input_sockets = {
		'Data': sockets.AnySocketDef(),
		'Freq': sockets.PhysicalFreqSocketDef(),
	}
	# input_sockets_sets: typ.ClassVar = {
	# '2D Freq': {
	# 'Data': sockets.AnySocketDef(),
	# 'Freq': sockets.PhysicalFreqSocketDef(),
	# },
	# }
	output_sockets: typ.ClassVar = {
		'Preview': sockets.AnySocketDef(),
	}

	managed_obj_defs: typ.ClassVar = {
		'plot': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLImage(name),
		),
		#'empty': ct.schemas.ManagedObjDef(
		# mk=lambda name: managed_objs.ManagedBLEmpty(name),
		# ),
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
		managed_objs: dict[str, ct.schemas.ManagedObj],
		input_sockets: dict,
		props: dict,
		unit_systems: dict,
	):
		selected_data = jnp.array(
			input_sockets['Data'].sel(f=input_sockets['Freq'], method='nearest')
		)

		managed_objs['plot'].xyzf_to_image(
			selected_data,
			colormap=props['colormap'],
			bl_select=True,
		)

	# @events.on_init()
	# def on_init(self):
	# self.on_changed_inputs()


####################
# - Blender Registration
####################
BL_REGISTER = [
	VizNode,
]
BL_NODES = {ct.NodeType.Viz: (ct.NodeCategory.MAXWELLSIM_ANALYSIS)}
