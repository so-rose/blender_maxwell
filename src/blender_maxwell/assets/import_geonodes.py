"""Provides for the linking and/or appending of geometry nodes trees from vendored libraries included in Blender maxwell."""

import enum
import typing as typ
from pathlib import Path

import bpy
import typing_extensions as typx

from .. import info
from blender_maxwell.utils import logger

log = logger.get(__name__)

ImportMethod: typ.TypeAlias = typx.Literal['append', 'link']
BLOperatorStatus: typ.TypeAlias = set[
	typx.Literal['RUNNING_MODAL', 'CANCELLED', 'FINISHED', 'PASS_THROUGH', 'INTERFACE']
]


####################
# - GeoNodes Specification
####################
class GeoNodes(enum.StrEnum):
	"""Defines available GeoNodes groups vendored as part of Blender Maxwell.

	The value of this StrEnum is both the name of the .blend file containing the GeoNodes group, and of the GeoNodes group itself.
	"""
	# Node Previews
	## Input
	InputConstantPhysicalPol = '_input_constant_physical_pol'
	## Source
	SourcePointDipole = '_source_point_dipole'
	SourcePlaneWave = '_source_plane_wave'
	SourceUniformCurrent = '_source_uniform_current'
	SourceTFSF = '_source_tfsf'
	SourceGaussianBeam = '_source_gaussian_beam'
	SourceAstigmaticGaussianBeam = '_source_astigmatic_gaussian_beam'
	SourceMode = '_source_mode'
	SourceEHArray = '_source_eh_array'
	SourceEHEquivArray = '_source_eh_equiv_array'
	## Structure
	StructurePrimitivePlane = '_structure_primitive_plane'
	StructurePrimitiveBox = '_structure_primitive_box'
	StructurePrimitiveSphere = '_structure_primitive_sphere'
	StructurePrimitiveCylinder = '_structure_primitive_cylinder'
	StructurePrimitiveRing = '_structure_primitive_ring'
	StructurePrimitiveCapsule = '_structure_primitive_capsule'
	StructurePrimitiveCone = '_structure_primitive_cone'
	## Monitor
	MonitorEHField = '_monitor_eh_field'
	MonitorPowerFlux = '_monitor_power_flux'
	MonitorEpsTensor = '_monitor_eps_tensor'
	MonitorDiffraction = '_monitor_diffraction'
	MonitorProjCartEHField = '_monitor_proj_eh_field'
	MonitorProjAngEHField = '_monitor_proj_ang_eh_field'
	MonitorProjKSpaceEHField = '_monitor_proj_k_space_eh_field'
	## Simulation
	SimulationSimDomain = '_simulation_sim_domain'
	SimulationBoundConds = '_simulation_bound_conds'
	SimulationBoundCondPML = '_simulation_bound_cond_pml'
	SimulationBoundCondPEC = '_simulation_bound_cond_pec'
	SimulationBoundCondPMC = '_simulation_bound_cond_pmc'
	SimulationBoundCondBloch = '_simulation_bound_cond_bloch'
	SimulationBoundCondPeriodic = '_simulation_bound_cond_periodic'
	SimulationBoundCondAbsorbing = '_simulation_bound_cond_absorbing'
	SimulationSimGrid = '_simulation_sim_grid'
	SimulationSimGridAxisAuto = '_simulation_sim_grid_axis_auto'
	SimulationSimGridAxisManual = '_simulation_sim_grid_axis_manual'
	SimulationSimGridAxisUniform = '_simulation_sim_grid_axis_uniform'
	SimulationSimGridAxisArray = '_simulation_sim_grid_axis_array'

	# Structures
	## Primitives
	PrimitiveBox = 'box'
	PrimitiveRing = 'ring'
	PrimitiveSphere = 'sphere'


# GeoNodes Paths
## Internal
GN_INTERNAL_PATH = info.PATH_ASSETS / 'internal' / 'primitives'
GN_INTERNAL_INPUTS_PATH = GN_INTERNAL_PATH / 'input'
GN_INTERNAL_SOURCES_PATH = GN_INTERNAL_PATH / 'source'
GN_INTERNAL_STRUCTURES_PATH = GN_INTERNAL_PATH / 'structure'
GN_INTERNAL_MONITORS_PATH = GN_INTERNAL_PATH / 'monitor'
GN_INTERNAL_SIMULATIONS_PATH = GN_INTERNAL_PATH / 'simulation'

## Structures
GN_STRUCTURES_PATH = info.PATH_ASSETS / 'structures'
GN_STRUCTURES_PRIMITIVES_PATH = GN_STRUCTURES_PATH / 'primitives'

GN_PARENT_PATHS: dict[GeoNodes, Path] = {
	# Node Previews
	## Input
	GeoNodes.InputConstantPhysicalPol: GN_INTERNAL_INPUTS_PATH,
	## Source
	GeoNodes.SourcePointDipole: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourcePlaneWave: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourceUniformCurrent: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourceTFSF: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourceGaussianBeam: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourceAstigmaticGaussianBeam: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourceMode: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourceEHArray: GN_INTERNAL_SOURCES_PATH,
	GeoNodes.SourceEHEquivArray: GN_INTERNAL_SOURCES_PATH,
	## Structure
	GeoNodes.StructurePrimitivePlane: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.StructurePrimitiveBox: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.StructurePrimitiveSphere: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.StructurePrimitiveCylinder: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.StructurePrimitiveRing: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.StructurePrimitiveCapsule: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.StructurePrimitiveCone: GN_INTERNAL_STRUCTURES_PATH,
	## Monitor
	GeoNodes.MonitorEHField: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.MonitorPowerFlux: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.MonitorEpsTensor: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.MonitorDiffraction: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.MonitorProjCartEHField: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.MonitorProjAngEHField: GN_INTERNAL_STRUCTURES_PATH,
	GeoNodes.MonitorProjKSpaceEHField: GN_INTERNAL_STRUCTURES_PATH,
	## Simulation
	GeoNodes.SimulationSimDomain: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationBoundConds: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationBoundCondPML: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationBoundCondPEC: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationBoundCondPMC: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationBoundCondBloch: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationBoundCondPeriodic: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationBoundCondAbsorbing: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationSimGrid: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationSimGridAxisAuto: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationSimGridAxisManual: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationSimGridAxisUniform: GN_INTERNAL_SIMULATIONS_PATH,
	GeoNodes.SimulationSimGridAxisArray: GN_INTERNAL_SIMULATIONS_PATH,

	# Structures
	GeoNodes.PrimitiveBox: GN_STRUCTURES_PRIMITIVES_PATH,
	GeoNodes.PrimitiveRing: GN_STRUCTURES_PRIMITIVES_PATH,
	GeoNodes.PrimitiveSphere: GN_STRUCTURES_PRIMITIVES_PATH,
}


####################
# - Import GeoNodes (Link/Append)
####################
def import_geonodes(
	geonodes: GeoNodes,
	import_method: ImportMethod,
) -> bpy.types.GeometryNodeGroup:
	"""Given a (name of a) GeoNodes group packaged with Blender Maxwell, link/append it to the current file, and return the node group.

	Parameters:
		geonodes: The (name of the) GeoNodes group, which ships with Blender Maxwell.
		import_method: Whether to link or append the GeoNodes group.
			When 'link', repeated calls will not link a new group; the existing group will simply be returned.

	Returns:
		A GeoNodes group available in the current .blend file, which can ex. be attached to a 'GeoNodes Structure' node.
	"""
	if (
		import_method == 'link'
		and geonodes in bpy.data.node_groups
	):
		return bpy.data.node_groups[geonodes]

	filename = geonodes
	filepath = str(
		GN_PARENT_PATHS[geonodes] / (geonodes + '.blend') / 'NodeTree' / geonodes
	)
	directory = filepath.removesuffix(geonodes)
	log.info(
		'%s GeoNodes (filename=%s, directory=%s, filepath=%s)',
		'Linking' if import_method == 'link' else 'Appending',
		filename,
		directory,
		filepath,
	)
	bpy.ops.wm.append(
		filepath=filepath,
		directory=directory,
		filename=filename,
		check_existing=False,
		set_fake=True,
		link=import_method == 'link',
	)

	return bpy.data.node_groups[geonodes]


####################
# - GeoNodes Asset Shelf Panel for MaxwellSimTree
####################
class NodeAssetPanel(bpy.types.Panel):
	bl_idname = 'blender_maxwell.panel__node_asset_panel'
	bl_label = 'Node GeoNodes Asset Panel'
	bl_space_type = 'NODE_EDITOR'
	bl_region_type = 'UI'
	bl_category = 'Assets'

	# @classmethod
	# def poll(cls, context):
	# return (
	# (space := context.get('space_data')) is not None
	# and (node_tree := space.get('node_tree')) is not None
	# and (node_tree.bl_idname == 'MaxwellSimTreeType')
	# )

	def draw(self, context):
		layout = self.layout
		workspace = context.workspace
		wm = context.window_manager

		# list_id must be unique otherwise behaviour gets weird when the template_asset_view is shown twice
		# (drag operator stops working in AssetPanelDrag, clickable area of all Assets in AssetPanelNoDrag gets
		# reduced to below the Asset name and clickable area of Current File Assets in AssetPanelDrag gets
		# reduced as if it didn't have a drag operator)
		_activate_op_props, _drag_op_props = layout.template_asset_view(
			'geo_nodes_asset_shelf',
			workspace,
			'asset_library_reference',
			wm,
			'active_asset_list',
			wm,
			'active_asset_index',
			drag_operator=AppendGeoNodes.bl_idname,
		)


####################
# - Append GeoNodes Operator
####################
def get_view_location(region, coords, ui_scale):
	x, y = region.view2d.region_to_view(*coords)
	return x / ui_scale, y / ui_scale


class AppendGeoNodes(bpy.types.Operator):
	"""Operator allowing the user to append a vendored GeoNodes tree for use in a simulation."""

	bl_idname = 'blender_maxwell.blends__import_geo_nodes'
	bl_label = 'Import GeoNode Tree'
	bl_description = 'Append a geometry node tree from the Blender Maxwell plugin, either via linking or appending'
	bl_options = frozenset({'REGISTER'})

	####################
	# - Properties
	####################
	_asset: bpy.types.AssetRepresentation | None = None

	####################
	# - UI
	####################
	def draw(self, _: bpy.types.Context) -> None:
		"""Draws the UI of the operator."""
		layout = self.layout
		col = layout.column()
		col.prop(self, 'geonodes_to_append', expand=True)

	####################
	# - Execution
	####################
	@classmethod
	def poll(cls, context: bpy.types.Context) -> bool:
		"""Defines when the operator can be run.

		Returns:
			Whether the operator can be run.
		"""
		return context.asset is not None

	def invoke(self, context: bpy.types.Context, _):
		return self.execute(context)

	def execute(self, context: bpy.types.Context) -> BLOperatorStatus:
		"""Initializes the while-dragging modal handler, which executes custom logic when the mouse button is released.

		Runs in response to drag_handler of a `UILayout.template_asset_view`.
		"""
		asset: bpy.types.AssetRepresentation = context.asset
		log.debug('Dragging Asset: %s', asset.name)

		# Store Asset for Modal & Drag Start
		self._asset = context.asset

		# Register Modal Operator & Tag Area for Redraw
		context.window_manager.modal_handler_add(self)
		context.area.tag_redraw()

		# Set Modal Cursor
		context.window.cursor_modal_set('CROSS')

		# Return Status of Running Modal
		return {'RUNNING_MODAL'}

	def modal(
		self, context: bpy.types.Context, event: bpy.types.Event
	) -> BLOperatorStatus:
		"""When LMB is released, creates a GeoNodes Structure node.

		Runs in response to events in the node editor while dragging an asset from the side panel.
		"""
		if (asset := self._asset) is None:
			return {'PASS_THROUGH'}

		if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
			log.debug('Released Dragged Asset: %s', asset.name)
			area = context.area
			editor_region = next(
				region for region in area.regions.values() if region.type == 'WINDOW'
			)

			# Check if Mouse Coordinates are:
			## - INSIDE of Node Editor
			## - INSIDE of Node Editor's WINDOW Region
			if (
				(event.mouse_x >= area.x and event.mouse_x < area.x + area.width)
				and (event.mouse_y >= area.y and event.mouse_y < area.y + area.height)
			) and (
				(
					event.mouse_x >= editor_region.x
					and event.mouse_x < editor_region.x + editor_region.width
				)
				and (
					event.mouse_y >= editor_region.y
					and event.mouse_y < editor_region.y + editor_region.height
				)
			):
				log.info(
					'Asset "%s" Released in Main Window of Node Editor', asset.name
				)
				space = context.space_data
				node_tree = space.node_tree

				# Computing GeoNodes View Location
				## 1. node_tree.cursor_location gives clicked loc, not released.
				## 2. event.mouse_region_* has inverted x wrt. event.mouse_*.
				##    - View2D.region_to_view expects the event.mouse_* order.
				##    - Is it a bug? Who knows!
				## 3. We compute it manually, to avoid the jank.
				node_location = get_view_location(
					editor_region,
					[
						event.mouse_x - editor_region.x,
						event.mouse_y - editor_region.y,
					],
					context.preferences.system.ui_scale,
				)

				# Create GeoNodes Structure Node
				## 1. Deselect other nodes
				## 2. Select the new one
				## 3. Move it into place
				## 4. Redraw (so we see the new node right away)
				log.info(
					'Creating GeoNodes Structure Node at (%d, %d)',
					*tuple(node_location),
				)
				bpy.ops.node.select_all(action='DESELECT')
				node = node_tree.nodes.new('GeoNodesStructureNodeType')
				node.select = True
				node.location.x = node_location[0]
				node.location.y = node_location[1]
				context.area.tag_redraw()

				# Import & Attach the GeoNodes Tree to the Node
				geonodes = import_geonodes(asset.name, 'append')
				node.inputs['GeoNodes'].value = geonodes

			# Restore the Pre-Modal Mouse Cursor Shape
			context.window.cursor_modal_restore()
			return {'FINISHED'}

		return {'RUNNING_MODAL'}


####################
# - Blender Registration
####################
# def initialize_asset_libraries(_: bpy.types.Scene):
# bpy.app.handlers.load_post.append(initialize_asset_libraries)
## TODO: Move to top-level registration.

asset_libraries = bpy.context.preferences.filepaths.asset_libraries
if (
	asset_library_idx := asset_libraries.find('Blender Maxwell')
) != -1 and asset_libraries['Blender Maxwell'].path != str(info.PATH_ASSETS):
	bpy.ops.preferences.asset_library_remove(asset_library_idx)

if 'Blender Maxwell' not in asset_libraries:
	bpy.ops.preferences.asset_library_add()
	asset_library = asset_libraries[-1]  ## Since the operator adds to the end
	asset_library.name = 'Blender Maxwell'
	asset_library.path = str(info.PATH_ASSETS)

bpy.types.WindowManager.active_asset_list = bpy.props.CollectionProperty(
	type=bpy.types.AssetHandle
)
bpy.types.WindowManager.active_asset_index = bpy.props.IntProperty()
## TODO: Do something differently

BL_REGISTER = [
	# GeoNodesAssetShelf,
	NodeAssetPanel,
	AppendGeoNodes,
]

BL_KEYMAP_ITEM_DEFS = [
	# {
	# '_': [
	# AppendGeoNodes.bl_idname,
	# 'LEFTMOUSE',
	# 'CLICK_DRAG',
	# ],
	# 'ctrl': False,
	# 'shift': False,
	# 'alt': False,
	# }
]
