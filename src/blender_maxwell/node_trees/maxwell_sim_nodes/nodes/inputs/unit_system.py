from ... import contracts as ct
from ... import sockets
from .. import base, events


class UnitSystemConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.UnitSystem
	bl_label = 'Unit System'

	input_sockets = {
		'Unit System': sockets.PhysicalUnitSystemSocketDef(
			show_by_default=True,
		),
	}
	output_sockets = {
		'Unit System': sockets.PhysicalUnitSystemSocketDef(),
	}

	####################
	# - Callbacks
	####################
	@events.computes_output_socket(
		'Unit System',
		input_sockets={'Unit System'},
	)
	def compute_unit_system(self, input_sockets) -> dict:
		return input_sockets['Unit System']


####################
# - Blender Registration
####################
BL_REGISTER = [
	UnitSystemConstantNode,
]
BL_NODES = {ct.NodeType.UnitSystem: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
