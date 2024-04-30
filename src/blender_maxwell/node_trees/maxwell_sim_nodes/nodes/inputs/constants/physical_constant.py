import enum
import typing as typ

import sympy as sp

from blender_maxwell.utils import bl_cache
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts, sockets
from ... import base, events


class PhysicalConstantNode(base.MaxwellSimTreeNode):
	"""A number of configurable unit dimension, ex. time, length, etc. .

	Attributes:
		physical_type: The physical type to specify.
		size: The size of the physical type, if it can be a vector.
	"""

	node_type = contracts.NodeType.PhysicalConstant
	bl_label = 'Physical Constant'

	input_sockets: typ.ClassVar = {
		'Value': sockets.ExprSocketDef(),
	}
	output_sockets: typ.ClassVar = {
		'Value': sockets.ExprSocketDef(),
	}

	####################
	# - Properties
	####################
	physical_type: spux.PhysicalType = bl_cache.BLField(
		spux.PhysicalType.Time,
		prop_ui=True,
	)

	mathtype: enum.Enum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_mathtypes(),
		prop_ui=True,
	)

	size: enum.Enum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_sizes(),
		prop_ui=True,
	)

	####################
	# - Searchers
	####################
	def search_mathtypes(self):
		return [
			mathtype.bl_enum_element(i)
			for i, mathtype in enumerate(self.physical_type.valid_mathtypes)
		]

	def search_sizes(self):
		return [
			spux.NumberSize1D.from_shape(shape).bl_enum_element(i)
			for i, shape in enumerate(self.physical_type.valid_shapes)
			if spux.NumberSize1D.supports_shape(shape)
		]

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name={'physical_type', 'mathtype', 'size'},
		props={'physical_type', 'mathtype', 'size'},
	)
	def on_mathtype_or_size_changed(self, props) -> None:
		"""Change the input/output expression sockets to match the mathtype and size declared in the node."""
		shape = spux.NumberSize1D(props['size']).shape

		# Set Input Socket Physical Type
		if self.inputs['Value'].physical_type != props['physical_type']:
			self.inputs['Value'].physical_type = props['physical_type']
			self.search_mathtypes = bl_cache.Signal.ResetEnumItems
			self.search_sizes = bl_cache.Signal.ResetEnumItems

		# Set Input Socket Math Type
		if self.inputs['Value'].mathtype != props['mathtype']:
			self.inputs['Value'].mathtype = props['mathtype']

		# Set Input Socket Shape
		if self.inputs['Value'].shape != shape:
			self.inputs['Value'].shape = shape

	####################
	# - Callbacks
	####################
	@events.computes_output_socket('value')
	def compute_value(self: contracts.NodeTypeProtocol) -> sp.Expr:
		return self.compute_input('value')


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalConstantNode,
]
BL_NODES = {
	contracts.NodeType.PhysicalConstant: (
		contracts.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS
	)
}
