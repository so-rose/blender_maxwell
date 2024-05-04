import typing as typ

import bpy

from blender_maxwell.utils import bl_cache
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts as ct
from .... import sockets
from ... import base, events


class NumberConstantNode(base.MaxwellSimNode):
	"""A unitless number of configurable math type ex. integer, real, etc. .

	Attributes:
		mathtype: The math type to specify the number as.
	"""

	node_type = ct.NodeType.NumberConstant
	bl_label = 'Numerical Constant'

	input_sockets: typ.ClassVar = {
		'Value': sockets.ExprSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Value': sockets.ExprSocketDef(),
	}

	####################
	# - Properties
	####################
	mathtype: spux.MathType = bl_cache.BLField(
		spux.MathType.Integer,
		prop_ui=True,
	)

	size: spux.NumberSize1D = bl_cache.BLField(
		spux.NumberSize1D.Scalar,
		prop_ui=True,
	)

	####################
	# - UI
	####################
	def draw_props(self, _, col: bpy.types.UILayout) -> None:
		row = col.row(align=True)
		row.prop(self, self.blfields['mathtype'], text='')
		row.prop(self, self.blfields['size'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(prop_name={'mathtype', 'size'}, props={'mathtype', 'size'})
	def on_mathtype_size_changed(self, props) -> None:
		"""Change the input/output expression sockets to match the mathtype declared in the node."""
		self.inputs['Value'].mathtype = props['mathtype']
		self.inputs['Value'].shape = props['size'].shape

	####################
	# - FlowKind
	####################
	@events.computes_output_socket('Value', input_sockets={'Value'})
	def compute_value(self, input_sockets) -> typ.Any:
		return input_sockets['Value']


####################
# - Blender Registration
####################
BL_REGISTER = [
	NumberConstantNode,
]
BL_NODES = {ct.NodeType.NumberConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)}
