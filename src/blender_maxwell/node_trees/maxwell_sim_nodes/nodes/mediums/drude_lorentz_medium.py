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

import sympy.physics.units as spu
import tidy3d as td

from ... import contracts, sockets
from .. import base, events


class DrudeLorentzMediumNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.DrudeLorentzMedium

	bl_label = 'Drude-Lorentz Medium'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = (
		{
			'eps_inf': sockets.RealNumberSocketDef(
				label='εr_∞',
			),
		}
		| {
			f'del_eps{i}': sockets.RealNumberSocketDef(
				label=f'Δεr_{i}',
			)
			for i in [1, 2, 3]
		}
		| {
			f'f{i}': sockets.PhysicalFreqSocketDef(
				label=f'f_{i}',
			)
			for i in [1, 2, 3]
		}
		| {
			f'delta{i}': sockets.PhysicalFreqSocketDef(
				label=f'δ_{i}',
			)
			for i in [1, 2, 3]
		}
	)
	output_sockets = {
		'medium': sockets.MaxwellMediumSocketDef(label='Medium'),
	}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket('medium')
	def compute_medium(self: contracts.NodeTypeProtocol) -> td.Sellmeier:
		## Retrieval
		return td.Lorentz(
			eps_inf=self.compute_input('eps_inf'),
			coeffs=[
				(
					self.compute_input(f'del_eps{i}'),
					spu.convert_to(
						self.compute_input(f'f{i}'),
						spu.hertz,
					)
					/ spu.hertz,
					spu.convert_to(
						self.compute_input(f'delta{i}'),
						spu.hertz,
					)
					/ spu.hertz,
				)
				for i in [1, 2, 3]
			],
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	DrudeLorentzMediumNode,
]
BL_NODES = {
	contracts.NodeType.DrudeLorentzMedium: (contracts.NodeCategory.MAXWELLSIM_MEDIUMS)
}
