import typing as typ

import bpy

from ......utils import sci_constants as constants
from .... import contracts as ct
from .... import sockets
from ... import base, events


class ScientificConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.ScientificConstant
	bl_label = 'Scientific Constant'

	output_sockets: typ.ClassVar = {
		'Value': sockets.AnySocketDef(),
	}

	####################
	# - Properties
	####################
	sci_constant: bpy.props.StringProperty(
		name='Sci Constant',
		description='The name of a scientific constant',
		default='',
		search=lambda self, _, edit_text: self.search_sci_constants(edit_text),
		update=lambda self, context: self.on_update_sci_constant(context),
	)

	cache__units: bpy.props.StringProperty(default='')
	cache__uncertainty: bpy.props.StringProperty(default='')

	def search_sci_constants(
		self,
		edit_text: str,
	):
		return [name for name in constants.SCI_CONSTANTS if edit_text in name]

	def on_update_sci_constant(
		self,
		context: bpy.types.Context,
	):
		if self.sci_constant:
			self.cache__units = str(
				constants.SCI_CONSTANTS_INFO[self.sci_constant]['units']
			)
			self.cache__uncertainty = str(
				constants.SCI_CONSTANTS_INFO[self.sci_constant]['uncertainty']
			)
		else:
			self.cache__units = ''
			self.cache__uncertainty = ''

		self.sync_prop('sci_constant', context)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		col.prop(self, 'sci_constant', text='')

	def draw_info(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		if self.sci_constant:
			col.label(text=f'Units: {self.cache__units}')
			col.label(text=f'Uncertainty: {self.cache__uncertainty}')

		col.label(text=f'Ref: {constants.SCI_CONSTANTS_REF[0]}')

	####################
	# - Callbacks
	####################
	@events.computes_output_socket('Value', props={'sci_constant'})
	def compute_value(self, props: dict) -> typ.Any:
		return constants.SCI_CONSTANTS[props['sci_constant']]


####################
# - Blender Registration
####################
BL_REGISTER = [
	ScientificConstantNode,
]
BL_NODES = {
	ct.NodeType.ScientificConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)
}
