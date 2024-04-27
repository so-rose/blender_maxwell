"""Provides for the linking and/or appending of geometry nodes trees from vendored libraries included in Blender maxwell."""

import enum
from pathlib import Path

import bpy

from blender_maxwell import contracts as ct
from blender_maxwell.utils import logger

log = logger.get(__name__)


####################
# - GeoNodes Specification
####################
# GeoNodes Paths
## Internal
GN_INTERNAL_PATH = ct.addon.PATH_ASSETS / 'internal'
GN_INTERNAL_INPUTS_PATH = GN_INTERNAL_PATH / 'input'
GN_INTERNAL_SOURCES_PATH = GN_INTERNAL_PATH / 'source'
GN_INTERNAL_STRUCTURES_PATH = GN_INTERNAL_PATH / 'structure'
GN_INTERNAL_MONITORS_PATH = GN_INTERNAL_PATH / 'monitor'
GN_INTERNAL_SIMULATIONS_PATH = GN_INTERNAL_PATH / 'simulation'

## Structures
GN_STRUCTURES_PATH = ct.addon.PATH_ASSETS / 'structures'
GN_STRUCTURES_PRIMITIVES_PATH = GN_STRUCTURES_PATH / 'primitives'
GN_STRUCTURES_ARRAYS_PATH = GN_STRUCTURES_PATH / 'arrays'


class GeoNodes(enum.StrEnum):
	"""Defines the names of available GeoNodes groups vendored as part of Blender Maxwell.

	The values define both name of the .blend file containing the GeoNodes group, and the GeoNodes group itself.
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
	PrimitiveSphere = 'sphere'
	PrimitiveCylinder = 'cylinder'
	PrimitiveRing = 'ring'
	## Arrays
	ArrayRing = 'array_ring'

	@property
	def dedicated_node_type(self) -> ct.BLImportMethod:
		"""Deduces the denode type that implements a vendored GeoNodes tree (usually just "GeoNodes Structure").

		Generally, "GeoNodes Structure' is the generic triangle-mesh node that can do everything.
		Indeed, one could make perfectly useful simulations exclusively using triangle meshes.

		However, when nodes that use geometry directly supported by `Tidy3D` might be more performant, and support features (ex. differentiable parameters) critical to some simulations.

		To bridge the gap, this method provides a regularized hard-coded method of getting whichever node type is most appropriate for the given structure.

		Warnings:
			**The strings be manually checked, statically**.
			Since we don't have easy access to the `contracts` within the sim node tree, these are just plain strings.

			It's not ideal, definitely a lazy workaround, but hey.

		Returns:
			The node type (as a string) that should be created to expose the GeoNodes group.
		"""
		dedicated_node_map = {
			GeoNodes.PrimitiveBox: 'BoxStructureNodeType',
			GeoNodes.PrimitiveSphere: 'SphereStructureNodeType',
			GeoNodes.PrimitiveCylinder: 'CylinderStructureNodeType',
		}

		if dedicated_node_map.get(self) is None:
			return 'GeoNodesStructureNodeType'

		return dedicated_node_map[self]

	@property
	def import_method(self) -> ct.BLImportMethod:
		"""Deduces whether a vendored GeoNodes tree should be linked or appended.

		Currently, everything is linked.
		If the user wants to modify a group, they can always make a local copy of a linked group and just use that.

		Returns:
			Either 'link' or 'append'.
		"""
		return 'link'

	@property
	def parent_path(self) -> Path:
		"""Deduces the parent path of a vendored GeoNodes tree.

		Warnings:
			The guarantee of same-named `.blend` and tree name are not explicitly checked.
			Also, the validity of GeoNodes tree interface parameters is also unchecked; this must be manually coordinated between the driving node, and the imported tree.

		Returns:
			The parent path of the given GeoNodes tree.

			A `.blend` file of the same name as the value of the enum must be **guaranteed** to exist as a direct child of the returned path, and must be **guaranteed** to contain a GeoNodes tree of the same name.
		"""
		GN = GeoNodes
		return {
			# Node Previews
			## Input
			GN.InputConstantPhysicalPol: GN_INTERNAL_INPUTS_PATH,
			## Source
			GN.SourcePointDipole: GN_INTERNAL_SOURCES_PATH,
			GN.SourcePlaneWave: GN_INTERNAL_SOURCES_PATH,
			GN.SourceUniformCurrent: GN_INTERNAL_SOURCES_PATH,
			GN.SourceTFSF: GN_INTERNAL_SOURCES_PATH,
			GN.SourceGaussianBeam: GN_INTERNAL_SOURCES_PATH,
			GN.SourceAstigmaticGaussianBeam: GN_INTERNAL_SOURCES_PATH,
			GN.SourceMode: GN_INTERNAL_SOURCES_PATH,
			GN.SourceEHArray: GN_INTERNAL_SOURCES_PATH,
			GN.SourceEHEquivArray: GN_INTERNAL_SOURCES_PATH,
			## Structure
			GN.StructurePrimitivePlane: GN_INTERNAL_STRUCTURES_PATH,
			GN.StructurePrimitiveBox: GN_INTERNAL_STRUCTURES_PATH,
			GN.StructurePrimitiveSphere: GN_INTERNAL_STRUCTURES_PATH,
			GN.StructurePrimitiveCylinder: GN_INTERNAL_STRUCTURES_PATH,
			GN.StructurePrimitiveRing: GN_INTERNAL_STRUCTURES_PATH,
			GN.StructurePrimitiveCapsule: GN_INTERNAL_STRUCTURES_PATH,
			GN.StructurePrimitiveCone: GN_INTERNAL_STRUCTURES_PATH,
			## Monitor
			GN.MonitorEHField: GN_INTERNAL_STRUCTURES_PATH,
			GN.MonitorPowerFlux: GN_INTERNAL_STRUCTURES_PATH,
			GN.MonitorEpsTensor: GN_INTERNAL_STRUCTURES_PATH,
			GN.MonitorDiffraction: GN_INTERNAL_STRUCTURES_PATH,
			GN.MonitorProjCartEHField: GN_INTERNAL_STRUCTURES_PATH,
			GN.MonitorProjAngEHField: GN_INTERNAL_STRUCTURES_PATH,
			GN.MonitorProjKSpaceEHField: GN_INTERNAL_STRUCTURES_PATH,
			## Simulation
			GN.SimulationSimDomain: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationBoundConds: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationBoundCondPML: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationBoundCondPEC: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationBoundCondPMC: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationBoundCondBloch: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationBoundCondPeriodic: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationBoundCondAbsorbing: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationSimGrid: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationSimGridAxisAuto: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationSimGridAxisManual: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationSimGridAxisUniform: GN_INTERNAL_SIMULATIONS_PATH,
			GN.SimulationSimGridAxisArray: GN_INTERNAL_SIMULATIONS_PATH,
			# Structures
			## Primitives
			GN.PrimitiveBox: GN_STRUCTURES_PRIMITIVES_PATH,
			GN.PrimitiveRing: GN_STRUCTURES_PRIMITIVES_PATH,
			GN.PrimitiveSphere: GN_STRUCTURES_PRIMITIVES_PATH,
			## Arrays
			GN.ArrayRing: GN_STRUCTURES_ARRAYS_PATH,
		}[self]


####################
# - Import GeoNodes (Link/Append)
####################
def import_geonodes(
	_geonodes: GeoNodes,
	force_append: bool = False,
) -> bpy.types.GeometryNodeGroup:
	"""Given vendored GeoNodes tree link/append and return the local datablock.

	- `GeoNodes.import_method` is used to determine whether it should be linked or imported.
	- `GeoNodes.parent_path` is used to determine

	Parameters:
		geonodes: The (name of the) GeoNodes group, which ships with Blender Maxwell.

	Returns:
		A GeoNodes group available in the current .blend file, which can ex. be attached to a 'GeoNodes Structure' node.
	"""
	# Parse Input
	geonodes = GeoNodes(_geonodes)
	import_method = geonodes.import_method

	# Linked: Don't Re-Link
	if import_method == 'link' and geonodes in bpy.data.node_groups:
		return bpy.data.node_groups[geonodes]

	filename = geonodes
	filepath = str(geonodes.parent_path / (geonodes + '.blend') / 'NodeTree' / geonodes)
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
ASSET_PANEL_NAME: str = 'blender_maxwell__node_asset_shelf'
WM_ACTIVE_NODE_ASSETS: str = 'blender_maxwell__active_node_asset_list'
WM_ACTIVE_ASSET_IDX: str = 'blender_maxwell__active_node_asset_index'


class NodeAssetPanel(bpy.types.Panel):
	"""Provides a panel that displays vendored, usable GeoNodes trees, and enables drag-and-drop support to easily drag them into the simulation node editor."""

	## TODO: Provide an option that forces appending? So that users can modify from a baseline. Just watch out - dealing with overlaps isn't trivial.

	bl_idname = ct.PanelType.NodeAssetPanel
	bl_label = 'Node GeoNodes Asset Panel'
	bl_space_type = 'NODE_EDITOR'
	bl_region_type = 'UI'
	bl_category = 'Assets'

	@classmethod
	def poll(cls, context: bpy.types.Context) -> bool:
		"""Require window manager properties to show node assets.

		Notes:
			Run by Blender when trying to show a panel.

		Returns:
			Whether the panel can show.
		"""
		## TODO: Check that we're in a MaxwellSim node tree as well.
		wm = context.window_manager
		return hasattr(wm, WM_ACTIVE_NODE_ASSETS) and hasattr(wm, WM_ACTIVE_ASSET_IDX)

	def draw(self, context: bpy.types.Context) -> None:
		"""Draw the asset library w/drag support.

		Notes:
			Run by Blender when the panel needs to be displayed.

		Parameters:
			context: The Blender context object.
				Must contain `context.window_manager` and `context.workspace`.
		"""
		layout = self.layout

		# Draw the Asset Panel w/Drag Support
		_activate_op_props, _drag_op_props = layout.template_asset_view(
			# list_id
			## Identifies this particular UI-bound asset list.
			## Must be unique, otherwise weird things happen if invoked shown twice.
			## We use a custom name, presuming only one tree is shown at a time.
			## TODO: Allow several trees to show at once.
			ASSET_PANEL_NAME,
			# asset_library_[dataptr|propname]
			## Where to find the asset library to pull assets from.
			## We pull it from a global
			context.workspace,
			'asset_library_reference',
			# assets_[dataptr|propname]
			## The actual assets to provide user access to.
			context.window_manager,
			WM_ACTIVE_NODE_ASSETS,
			# assets_[dataptr|propname]
			## The currently selected asset to highlight.
			context.window_manager,
			WM_ACTIVE_ASSET_IDX,
			# draw_operator
			## The operator to invoke() whenever the user **starts** dragging an asset.
			drag_operator=ct.OperatorType.GeoNodesToStructureNode,
			# Other Options
			## The currently selected asset to highlight.
			# display_options={'NO_LIBRARY'},
		)


####################
# - Append GeoNodes Operator
####################
def get_view_location(
	region: bpy.types.Region, coords: tuple[float, float], ui_scale: float
) -> tuple[float, float]:
	"""Given a pair of coordinates defined on a Blender region, project the coordinates to the corresponding `bpy.types.View2D`.

	Parameters:
		region: The region within which the coordinates are defined.
		coords: The coordinates within the region, ex. produced during a mouse click event.
		ui_scale: Scaling factor applied to pixels, which compensates for screens with differing DPIs.
			We must divide the coordinates we receive by this float to get the "real" $x,y$ coordinates.

	Returns:
		The $x,y$ coordinates within the region's `bpy.types.View2D`, correctly scaled with the current interface size/dpi.
	"""
	x, y = region.view2d.region_to_view(*coords)
	return x / ui_scale, y / ui_scale


class GeoNodesToStructureNode(bpy.types.Operator):
	"""Operator allowing the user to append a vendored GeoNodes tree for use in a simulation."""

	bl_idname = ct.OperatorType.GeoNodesToStructureNode
	bl_label = 'GeoNodes to Structure Node'
	bl_description = 'Drag-and-drop operator'
	bl_options = frozenset({'REGISTER'})

	@classmethod
	def poll(cls, context: bpy.types.Context) -> bool:
		"""Defines when the operator can be run.

		Returns:
			Whether the operator can be run.
		"""
		return context.asset is not None

	####################
	# - Properties
	####################
	_asset: bpy.types.AssetRepresentation | None = None

	####################
	# - Execution
	####################
	def invoke(
		self, context: bpy.types.Context, _: bpy.types.Event
	) -> set[ct.BLOperatorStatus]:
		"""Commences the drag-and-drop of a GeoNodes asset.

		- Starts the modal timer, which listens for a mouse-release event.
		- Changes the mouse cursor to communicate the "dragging" state to the user.

		Notes:
			Run via `NodeAssetPanel` when the user starts dragging a vendored GeoNodes asset.

			The asset being dragged can be found in `context.asset`.

		Returns:
			Indication that there is a modal running.
		"""
		self._asset = context.asset  ## Guaranteed non-None by poll()

		# Register Modal Operator
		context.window_manager.modal_handler_add(self)
		context.area.tag_redraw()

		# Set Mouse Cursor
		context.window.cursor_modal_set('CROSS')

		return {'RUNNING_MODAL'}

	def modal(
		self, context: bpy.types.Context, event: bpy.types.Event
	) -> ct.BLOperatorStatus:
		"""When LMB is released, creates a GeoNodes Structure node.

		Runs in response to events in the node editor while dragging an asset from the side panel.
		"""
		# No Asset: Do Nothing
		asset = self._asset
		if asset is None:
			return {'PASS_THROUGH'}

		# Released LMB: Add Structure Node
		## The user 'dropped' the asset!
		## Now, we create an appropriate "Structure" node corresponding to the dropped asset.
		if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
			log.debug('Dropped Asset: %s', asset.name)

			# Load Editor Region
			area = context.area
			editor_window_regions = [
				region for region in area.regions.values() if region.type == 'WINDOW'
			]
			if editor_window_regions:
				log.debug(
					'Selected Node Editor Region out of %s Options',
					str(len(editor_window_regions)),
				)
				editor_region = editor_window_regions[0]
			else:
				msg = 'No valid editor region in context where asset was dropped'
				raise RuntimeError(msg)

			# Check if Mouse Coordinates are:
			## - INSIDE of Node Editor
			## - INSIDE of Node Editor's WINDOW Region
			# Check Mouse Coordinates against Node Editor
			## The asset must be dropped inside the Maxwell Sim node editor.
			if (
				## Check: Mouse in Area (contextual)
				(event.mouse_x >= area.x and event.mouse_x < area.x + area.width)
				and (event.mouse_y >= area.y and event.mouse_y < area.y + area.height)
			) and (
				## Check: Mouse in Node Editor Region
				(
					event.mouse_x >= editor_region.x
					and event.mouse_x < editor_region.x + editor_region.width
				)
				and (
					event.mouse_y >= editor_region.y
					and event.mouse_y < editor_region.y + editor_region.height
				)
			):
				space = context.space_data
				node_tree = space.node_tree

				# Computing GeoNodes View Location
				## 1. node_tree.cursor_location gives clicked loc, not released.
				## 2. event.mouse_region_* has inverted x wrt. event.mouse_*.
				##    -> View2D.region_to_view expects the event.mouse_* order.
				##    -> Is it a bug? Who knows!
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
				geonodes = GeoNodes(asset.name)

				log.info(
					'Creating Node "%s" at (%d, %d)',
					geonodes.dedicated_node_type,
					*tuple(node_location),
				)
				bpy.ops.node.select_all(action='DESELECT')
				node = node_tree.nodes.new(geonodes.dedicated_node_type)
				node.select = True
				node.location.x = node_location[0]
				node.location.y = node_location[1]
				context.area.tag_redraw()

				# GeoNodes Structure Specific Setup
				## Since the node doesn't itself handle the structure, we must.
				## We just import the GN tree, then attach the data block to the node.
				if geonodes.dedicated_node_type == 'GeoNodesStructureNodeType':
					geonodes_data = import_geonodes(asset.name)
					node.inputs['GeoNodes'].value = geonodes_data
					## TODO: Is this too presumptuous? Or a fine little hack?

			# Restore the Pre-Modal Mouse Cursor Shape
			context.window.cursor_modal_restore()
			return {'FINISHED'}

		return {'RUNNING_MODAL'}


####################
# - Blender Registration
####################
ASSET_LIB_POSTFIX: str = ' | BLMaxwell'
ASSET_LIB_SPECS: dict[str, Path] = {
	'Primitives': GN_STRUCTURES_PRIMITIVES_PATH,
	'Arrays': GN_STRUCTURES_ARRAYS_PATH,
}


@bpy.app.handlers.persistent
def initialize_asset_libraries(_: bpy.types.Scene):
	"""Before loading a `.blend` file, ensure that the WindowManager properties relied on by NodeAssetPanel are available.

	- Several asset libraries, defined under the global `ASSET_LIB_SPECS`, are added/replaced such that the name:path map is respected.
	- Existing asset libraries are left entirely alone.
	- The names attached to `bpy.types.WindowManager` are specifically the globals `WM_ACTIVE_NODE_ASSETS` and `WM_ACTIVE_ASSET_IDX`.

	Warnings:
		**Changing the name of an asset library will leave it dangling on all `.blend` files**.

		Therefore, an addon-specific postfix `ASSET_LIB_POSTFIX` is appended to all asset library names.
		**This postfix must never, ever change, across any and all versions of the addon**.

		Again, if it does, **all Blender installations having ever used the addon will have a dangling asset library until the user manually notices + removes it**.
	"""
	asset_libraries = bpy.context.preferences.filepaths.asset_libraries

	# Guarantee All Asset Libraries
	for _asset_lib_name, asset_lib_path in ASSET_LIB_SPECS.items():
		asset_lib_name = _asset_lib_name + ASSET_LIB_POSTFIX

		# Remove Existing Asset Library
		asset_library_idx = asset_libraries.find(asset_lib_name)
		if asset_library_idx != -1 and asset_libraries[asset_lib_name].path != str(
			asset_lib_path
		):
			log.debug('Removing Asset Library: %s', asset_lib_name)
			bpy.ops.preferences.asset_library_remove(asset_library_idx)

		# Add Asset Library
		if asset_lib_name not in asset_libraries:
			log.debug('Add Asset Library: %s', asset_lib_name)

			bpy.ops.preferences.asset_library_add()
			asset_library = asset_libraries[-1]  ## Was added to the end
			asset_library.name = asset_lib_name
			asset_library.path = str(asset_lib_path)

	# Set WindowManager Props
	## Set Active Assets Collection
	setattr(
		bpy.types.WindowManager,
		WM_ACTIVE_NODE_ASSETS,
		bpy.props.CollectionProperty(type=bpy.types.AssetHandle),
	)
	## Set Active Assets Collection
	setattr(
		bpy.types.WindowManager,
		WM_ACTIVE_ASSET_IDX,
		bpy.props.IntProperty(),
	)


bpy.app.handlers.load_pre.append(initialize_asset_libraries)

BL_REGISTER = [
	NodeAssetPanel,
	GeoNodesToStructureNode,
]

BL_HOTKEYS = []
