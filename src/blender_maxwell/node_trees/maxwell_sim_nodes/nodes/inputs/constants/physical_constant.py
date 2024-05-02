import typing as typ

import bpy
import sympy as sp

from blender_maxwell.utils import bl_cache
from blender_maxwell.utils import extra_sympy_units as spux

from .... import contracts, sockets
from ... import base, events


class PhysicalConstantNode(base.MaxwellSimNode):
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

	mathtype: spux.MathType = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_mathtypes(),
		prop_ui=True,
	)

	size: spux.NumberSize1D = bl_cache.BLField(
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
	# - UI
	####################
	def draw_props(self, _, col: bpy.types.UILayout) -> None:
		col.prop(self, self.blfields['physical_type'], text='')

		row = col.row(align=True)
		row.prop(self, self.blfields['mathtype'], text='')
		row.prop(self, self.blfields['size'], text='')

	####################
	# - Events
	####################
	@events.on_value_changed(
		# Trigger
		prop_name={'physical_type'},
		run_on_init=True,
		# Loaded
		props={'physical_type'},
	)
	def on_physical_type_changed(self, props) -> None:
		"""Change the input/output expression sockets to match the mathtype and size declared in the node."""
		# Set Input Socket Physical Type
		if self.inputs['Value'].physical_type != props['physical_type']:
			self.inputs['Value'].physical_type = props['physical_type']
			self.mathtype = bl_cache.Signal.ResetEnumItems
			self.size = bl_cache.Signal.ResetEnumItems

	@events.on_value_changed(
		# Trigger
		prop_name={'mathtype', 'size'},
		run_on_init=True,
		# Loaded
		props={'physical_type', 'mathtype', 'size'},
	)
	def on_mathtype_or_size_changed(self, props) -> None:
		# Set Input Socket Math Type
		if self.inputs['Value'].mathtype != props['mathtype']:
			self.inputs['Value'].mathtype = props['mathtype']

		# Set Input Socket Shape
		shape = props['size'].shape
		if self.inputs['Value'].shape != shape:
			self.inputs['Value'].shape = shape

	####################
	# - Callbacks
	####################
	@events.computes_output_socket('Value', input_sockets={'Value'})
	def compute_value(self, input_sockets) -> sp.Expr:
		return input_sockets['Value']


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
