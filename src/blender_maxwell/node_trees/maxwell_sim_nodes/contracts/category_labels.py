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

from .category_types import NodeCategory as NC  # noqa: N817

NODE_CAT_LABELS = {
	# Analysis/
	NC.MAXWELLSIM_ANALYSIS: 'Analysis',
	NC.MAXWELLSIM_ANALYSIS_MATH: 'Math',
	# Utilities/
	NC.MAXWELLSIM_UTILITIES: 'Utilities',
	# Inputs/
	NC.MAXWELLSIM_INPUTS: 'Inputs',
	NC.MAXWELLSIM_INPUTS_SCENE: 'Scene',
	NC.MAXWELLSIM_INPUTS_CONSTANTS: 'Constants',
	NC.MAXWELLSIM_INPUTS_FILEIMPORTERS: 'File Importers',
	NC.MAXWELLSIM_INPUTS_WEBIMPORTERS: 'Web Importers',
	# Solvers/
	NC.MAXWELLSIM_SOLVERS: 'Solvers',
	# Outputs/
	NC.MAXWELLSIM_OUTPUTS: 'Outputs',
	NC.MAXWELLSIM_OUTPUTS_FILEEXPORTERS: 'File Exporters',
	NC.MAXWELLSIM_OUTPUTS_WEBEXPORTERS: 'Web Exporters',
	# Sources/
	NC.MAXWELLSIM_SOURCES: 'Sources',
	NC.MAXWELLSIM_SOURCES_TEMPORALSHAPES: 'Temporal Shapes',
	# Mediums/
	NC.MAXWELLSIM_MEDIUMS: 'Mediums',
	NC.MAXWELLSIM_MEDIUMS_NONLINEARITIES: 'Non-Linearities',
	# Structures/
	NC.MAXWELLSIM_STRUCTURES: 'Structures',
	NC.MAXWELLSIM_STRUCTURES_PRIMITIVES: 'Primitives',
	# Monitors/
	NC.MAXWELLSIM_MONITORS: 'Monitors',
	NC.MAXWELLSIM_MONITORS_PROJECTED: 'Projected',
	# Simulations/
	NC.MAXWELLSIM_SIMS: 'Simulations',
	NC.MAXWELLSIM_SIMS_BOUNDCONDFACES: 'BC Faces',
	NC.MAXWELLSIM_SIMS_SIMGRIDAXES: 'Grid Axes',
}
