import sympy.physics.units as spu
import tidy3d as td

from ... import contracts, sockets
from .. import base, events


class TripleSellmeierMediumNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.TripleSellmeierMedium

	bl_label = 'Three-Parameter Sellmeier Medium'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {
		f'B{i}': sockets.RealNumberSocketDef(
			label=f'B{i}',
		)
		for i in [1, 2, 3]
	} | {
		f'C{i}': sockets.PhysicalAreaSocketDef(label=f'C{i}', default_unit=spu.um**2)
		for i in [1, 2, 3]
	}
	output_sockets = {
		'medium': sockets.MaxwellMediumSocketDef(label='Medium'),
	}

	####################
	# - Presets
	####################
	presets = {
		'BK7': contracts.PresetDef(
			label='BK7 Glass',
			description='Borosilicate crown glass (known as BK7)',
			values={
				'B1': 1.03961212,
				'B2': 0.231792344,
				'B3': 1.01046945,
				'C1': 6.00069867e-3 * spu.um**2,
				'C2': 2.00179144e-2 * spu.um**2,
				'C3': 103.560653 * spu.um**2,
			},
		),
		'FUSED_SILICA': contracts.PresetDef(
			label='Fused Silica',
			description='Fused silica aka. SiO2',
			values={
				'B1': 0.696166300,
				'B2': 0.407942600,
				'B3': 0.897479400,
				'C1': 4.67914826e-3 * spu.um**2,
				'C2': 1.35120631e-2 * spu.um**2,
				'C3': 97.9340025 * spu.um**2,
			},
		),
	}

	####################
	# - Output Socket Computation
	####################
	@events.computes_output_socket('medium')
	def compute_medium(self: contracts.NodeTypeProtocol) -> td.Sellmeier:
		## Retrieval
		# B1 = self.compute_input("B1")
		# C1_with_units = self.compute_input("C1")
		#
		## Processing
		# C1 = spu.convert_to(C1_with_units, spu.um**2) / spu.um**2

		return td.Sellmeier(
			coeffs=[
				(
					self.compute_input(f'B{i}'),
					spu.convert_to(
						self.compute_input(f'C{i}'),
						spu.um**2,
					)
					/ spu.um**2,
				)
				for i in [1, 2, 3]
			]
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	TripleSellmeierMediumNode,
]
BL_NODES = {
	contracts.NodeType.TripleSellmeierMedium: (
		contracts.NodeCategory.MAXWELLSIM_MEDIUMS
	)
}
