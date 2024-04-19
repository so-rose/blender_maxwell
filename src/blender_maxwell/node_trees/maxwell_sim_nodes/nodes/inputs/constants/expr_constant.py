import typing as typ

from .... import contracts as ct
from .... import sockets
from ... import base, events


class ExprConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.ExprConstant
	bl_label = 'Expr Constant'

	input_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Expr': sockets.ExprSocketDef(),
	}

	## TODO: Symbols (defined w/props?)
	## - Currently expr constant isn't excessively useful, since there are no variables.
	## - We'll define the #, type, name with props.
	## - We'll add loose-socket inputs as int/real/complex/physical socket (based on type) for Param.
	## - We the output expr would support `Value` (just the expression), `LazyValueFunc` (evaluate w/symbol support), `Param` (example values for symbols).

	####################
	# - Callbacks
	####################
	@events.computes_output_socket(
		'Expr', kind=ct.FlowKind.Value, input_sockets={'Expr'}
	)
	def compute_value(self, input_sockets: dict) -> typ.Any:
		return input_sockets['Expr']


####################
# - Blender Registration
####################
BL_REGISTER = [
	ExprConstantNode,
]
BL_NODES = {ct.NodeType.ExprConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)}
