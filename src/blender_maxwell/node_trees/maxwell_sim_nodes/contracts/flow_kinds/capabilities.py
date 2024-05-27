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

import dataclasses
import typing as typ
from types import MappingProxyType

from ..socket_types import SocketType
from .flow_kinds import FlowKind


@dataclasses.dataclass(frozen=True, kw_only=True)
class CapabilitiesFlow:
	"""Describes the compatibility relationship between two sockets, which governs whether they can be linked.

	By default, socket type (which may impact color) and active `FlowKind` (which impacts shape) must match in order for two sockets to be compatible.

	However, in many cases, more expressiveness beyond this naive constraint is desirable.
	For example:

	- Allow any socket to be linked to the `ViewerNode` input.
	- Allow only _angled_ sources to be passed as inputs to the input-derived `BlochBoundCond` node.
	- Allow `Expr:Value` to connect to `Expr:Func`, but only allow the converse if `PhysicalType`, `MathType`, and `Size` match.

	In many cases, it's desirable

	"""

	# Defaults
	socket_type: SocketType
	active_kind: FlowKind

	# Relationships
	allow_out_to_in: dict[FlowKind, FlowKind] = dataclasses.field(default_factory=dict)
	allow_out_to_in_if_matches: dict[FlowKind, (FlowKind, bool)] = dataclasses.field(
		default_factory=dict
	)

	is_universal: bool = False

	# == Constraint
	must_match: dict[str, typ.Any] = dataclasses.field(default_factory=dict)

	# ∀b (b ∈ A) Constraint
	## A: allow_any
	## b∈B: present_any
	allow_any: set[typ.Any] = dataclasses.field(default_factory=set)
	present_any: set[typ.Any] = dataclasses.field(default_factory=set)

	def is_compatible_with(self, other: typ.Self) -> bool:
		return other.is_universal or (
			self.socket_type is other.socket_type
			and (
				self.active_kind is other.active_kind
				or (
					other.active_kind in other.allow_out_to_in
					and self.active_kind is other.allow_out_to_in[other.active_kind]
				)
				or (
					other.active_kind in other.allow_out_to_in_if_matches
					and self.active_kind
					is other.allow_out_to_in_if_matches[other.active_kind][0]
					and self.allow_out_to_in_if_matches[other.active_kind][1]
					== other.allow_out_to_in_if_matches[other.active_kind][1]
				)
			)
			# == Constraint
			and all(
				name in other.must_match
				and self.must_match[name] == other.must_match[name]
				for name in self.must_match
			)
			# ∀b (b ∈ A) Constraint
			and (
				self.present_any & other.allow_any
				or (not self.present_any and not self.allow_any)
			)
		)
