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

import enum

from blender_maxwell.utils import blender_type_enum


@blender_type_enum.wrap_values_in_MT
class NodeCategory(blender_type_enum.BlenderTypeEnum):
	MAXWELLSIM = enum.auto()

	# Analysis/
	MAXWELLSIM_ANALYSIS = enum.auto()
	MAXWELLSIM_ANALYSIS_MATH = enum.auto()

	# Utilities/
	MAXWELLSIM_UTILITIES = enum.auto()

	# Inputs/
	MAXWELLSIM_INPUTS = enum.auto()
	MAXWELLSIM_INPUTS_SCENE = enum.auto()
	MAXWELLSIM_INPUTS_CONSTANTS = enum.auto()
	MAXWELLSIM_INPUTS_FILEIMPORTERS = enum.auto()
	MAXWELLSIM_INPUTS_WEBIMPORTERS = enum.auto()

	# Solvers/
	MAXWELLSIM_SOLVERS = enum.auto()

	# Outputs/
	MAXWELLSIM_OUTPUTS = enum.auto()
	MAXWELLSIM_OUTPUTS_FILEEXPORTERS = enum.auto()
	MAXWELLSIM_OUTPUTS_WEBEXPORTERS = enum.auto()

	# Sources/
	MAXWELLSIM_SOURCES = enum.auto()
	MAXWELLSIM_SOURCES_TEMPORALSHAPES = enum.auto()

	# Mediums/
	MAXWELLSIM_MEDIUMS = enum.auto()
	MAXWELLSIM_MEDIUMS_NONLINEARITIES = enum.auto()

	# Structures/
	MAXWELLSIM_STRUCTURES = enum.auto()
	MAXWELLSIM_STRUCTURES_PRIMITIVES = enum.auto()

	# Monitors/
	MAXWELLSIM_MONITORS = enum.auto()
	MAXWELLSIM_MONITORS_PROJECTED = enum.auto()

	# Simulations/
	MAXWELLSIM_SIMS = enum.auto()
	MAXWELLSIM_SIMS_BOUNDCONDFACES = enum.auto()
	MAXWELLSIM_SIMS_SIMGRIDAXES = enum.auto()

	@classmethod
	def get_tree(cls):
		## TODO: Refactor
		syllable_categories = [
			str(node_category.value).split('_')
			for node_category in cls
			if node_category.value != 'MAXWELLSIM'
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
