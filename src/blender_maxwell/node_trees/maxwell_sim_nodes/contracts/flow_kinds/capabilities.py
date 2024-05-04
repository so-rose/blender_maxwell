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
