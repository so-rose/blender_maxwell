import sympy.physics.units as spu
import tidy3d as td

from ... import contracts, sockets
from .. import base


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
	@base.computes_output_socket('medium')
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
	contracts.NodeType.DrudeLorentzMedium: (
		contracts.NodeCategory.MAXWELLSIM_MEDIUMS
	)
}
