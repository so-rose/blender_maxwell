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
