import enum

from ....utils.blender_type_enum import (
	BlenderTypeEnum, wrap_values_in_MT
)

@wrap_values_in_MT
class NodeCategory(BlenderTypeEnum):
	MAXWELLSIM = enum.auto()
	
	# Inputs/
	MAXWELLSIM_INPUTS = enum.auto()
	MAXWELLSIM_INPUTS_IMPORTERS = enum.auto()
	MAXWELLSIM_INPUTS_SCENE = enum.auto()
	MAXWELLSIM_INPUTS_PARAMETERS = enum.auto()
	MAXWELLSIM_INPUTS_CONSTANTS = enum.auto()
	MAXWELLSIM_INPUTS_LISTS = enum.auto()
	
	# Outputs/
	MAXWELLSIM_OUTPUTS = enum.auto()
	MAXWELLSIM_OUTPUTS_VIEWERS = enum.auto()
	MAXWELLSIM_OUTPUTS_EXPORTERS = enum.auto()
	MAXWELLSIM_OUTPUTS_PLOTTERS = enum.auto()
	
	# Sources/
	MAXWELLSIM_SOURCES = enum.auto()
	MAXWELLSIM_SOURCES_TEMPORALSHAPES = enum.auto()
	
	# Mediums/
	MAXWELLSIM_MEDIUMS = enum.auto()
	MAXWELLSIM_MEDIUMS_NONLINEARITIES = enum.auto()
	
	# Structures/
	MAXWELLSIM_STRUCTURES = enum.auto()
	MAXWELLSIM_STRUCTURES_PRIMITIVES = enum.auto()
	
	# Bounds/
	MAXWELLSIM_BOUNDS = enum.auto()
	MAXWELLSIM_BOUNDS_BOUNDFACES = enum.auto()
	
	# Monitors/
	MAXWELLSIM_MONITORS = enum.auto()
	MAXWELLSIM_MONITORS_NEARFIELDPROJECTIONS = enum.auto()
	
	# Simulations/
	MAXWELLSIM_SIMS = enum.auto()
	MAXWELLSIM_SIMGRIDAXES = enum.auto()
	
	# Utilities/
	MAXWELLSIM_UTILITIES = enum.auto()
	MAXWELLSIM_UTILITIES_CONVERTERS = enum.auto()
	MAXWELLSIM_UTILITIES_OPERATIONS = enum.auto()
	
	@classmethod
	def get_tree(cls):
		## TODO: Refactor
		syllable_categories = [
			str(node_category.value).split("_")
			for node_category in cls
			if node_category.value != "MAXWELLSIM"
		]
		
		category_tree = {}
		for syllable_category in syllable_categories:
			# Set Current Subtree to Root
			current_category_subtree = category_tree
			
			for i, syllable in enumerate(syllable_category):
				# Create New Category Subtree and/or Step to Subtree
				if syllable not in current_category_subtree:
					current_category_subtree[syllable] = {}
				current_category_subtree = current_category_subtree[syllable]
		
		return category_tree
