import nodeitems_utils
from . import types

####################
# - Node Category Types
####################
CATEGORIES_MAXWELL_SIM = 'MAXWELL_SIM'

CATEGORY_MAXWELL_SIM_DEBUG = 'MAXWELL_SIM_DEBUG'
CATEGORY_MAXWELL_SIM_SOURCES = 'MAXWELL_SIM_SOURCES'
CATEGORY_MAXWELL_SIM_MEDIUMS = 'MAXWELL_SIM_MEDIUMS'
CATEGORY_MAXWELL_SIM_STRUCTURES = 'MAXWELL_SIM_STRUCTURES'
CATEGORY_MAXWELL_SIM_BOUNDS = 'MAXWELL_SIM_BOUNDS'
CATEGORY_MAXWELL_SIM_SIMULATIONS = 'MAXWELL_SIM_SIMULATIONS'

####################
# - Node Category Class
####################
class MaxwellSimNodeCategory(nodeitems_utils.NodeCategory):
	@classmethod
	def poll(cls, context):
		"""Constrain node category availability to within a MaxwellSimTree."""
		
		return context.space_data.tree_type == types.tree_types.MaxwellSimTreeType


####################
# - Node Category Definition
####################
CATEGORIES_MaxwellSimTree = [
	MaxwellSimNodeCategory(CATEGORY_MAXWELL_SIM_DEBUG, "Debug", items=[
		nodeitems_utils.NodeItem(types.DebugPrinterNodeType),
	]),
	MaxwellSimNodeCategory(CATEGORY_MAXWELL_SIM_SOURCES, "Sources", items=[
		nodeitems_utils.NodeItem(types.PointDipoleMaxwellSourceNodeType),
	]),
	MaxwellSimNodeCategory(CATEGORY_MAXWELL_SIM_MEDIUMS, "Mediums", items=[
		nodeitems_utils.NodeItem(types.SellmeierMaxwellMediumNodeType),
	]),
	MaxwellSimNodeCategory(CATEGORY_MAXWELL_SIM_STRUCTURES, "Structures", items=[
		nodeitems_utils.NodeItem(types.TriMeshMaxwellStructureNodeType),
	]),
	MaxwellSimNodeCategory(CATEGORY_MAXWELL_SIM_BOUNDS, "Bounds", items=[
		nodeitems_utils.NodeItem(types.PMLMaxwellBoundNodeType),
	]),
	MaxwellSimNodeCategory(CATEGORY_MAXWELL_SIM_SIMULATIONS, "Simulation", items=[
		nodeitems_utils.NodeItem(types.FDTDMaxwellSimulationNodeType),
	]),
]



####################
# - Blender Registration
####################
BL_NODE_CATEGORIES = [
	(CATEGORIES_MAXWELL_SIM, CATEGORIES_MaxwellSimTree)
]
