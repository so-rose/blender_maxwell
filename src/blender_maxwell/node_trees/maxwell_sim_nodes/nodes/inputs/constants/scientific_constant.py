import typing as typ

import bpy

from blender_maxwell.utils import bl_cache, sci_constants

from .... import contracts as ct
from .... import sockets
from ... import base, events


class ScientificConstantNode(base.MaxwellSimNode):
	node_type = ct.NodeType.ScientificConstant
	bl_label = 'Scientific Constant'

	output_sockets: typ.ClassVar = {
		'Value': sockets.ExprSocketDef(),
	}

	####################
	# - Properties
	####################
	sci_constant: str = bl_cache.BLField(
		'',
		prop_ui=True,
		str_cb=lambda self, _, edit_text: self.search_sci_constants(edit_text),
	)

	def search_sci_constants(
		self,
		edit_text: str,
	):
		return [
			name
			for name in sci_constants.SCI_CONSTANTS
			if edit_text.lower() in name.lower()
		]

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		col.prop(self, self.blfields['sci_constant'], text='')

	def draw_info(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		if self.sci_constant:
			col.label(
				text=f'Units: {sci_constants.SCI_CONSTANTS_INFO[self.sci_constant]["units"]}'
			)
			col.label(
				text=f'Uncertainty: {sci_constants.SCI_CONSTANTS_INFO[self.sci_constant]["uncertainty"]}'
			)

	####################
	# - Output
	####################
	@events.computes_output_socket('Value', props={'sci_constant'})
	def compute_value(self, props: dict) -> typ.Any:
		return sci_constants.SCI_CONSTANTS[props['sci_constant']]


####################
# - Blender Registration
####################
BL_REGISTER = [
	ScientificConstantNode,
]
BL_NODES = {
	ct.NodeType.ScientificConstant: (ct.NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS)
}
