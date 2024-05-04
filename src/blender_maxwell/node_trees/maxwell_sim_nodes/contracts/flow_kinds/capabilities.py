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

from ..socket_types import SocketType
from .flow_kinds import FlowKind


@dataclasses.dataclass(frozen=True, kw_only=True)
class CapabilitiesFlow:
	socket_type: SocketType
	active_kind: FlowKind

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
			self.socket_type == other.socket_type
			and self.active_kind == other.active_kind
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
