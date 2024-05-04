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

import enum
import typing as typ

_FLOW_SIGNAL_SET: set | None = None


class FlowSignal(enum.StrEnum):
	"""Special output socket return value, which indicates a piece of information about the state of the flow, instead of data.

	Attributes:
		FlowPending: The data that was requested is not available, but it is expected to become available soon.
			- **Behavior**: When encountered downstream in `events` decorators for all `FlowKind`s, either `FlowSignal.FlowPending` should return instead of the method, or the decorated method should simply not run.
			- **Caching**: Don't invalidate caches, since the user will expect their data to persist.
			- **Net Effect**: All nodes that encounter FlowPending are forward-locked, possibly with an explanatory indicator. In terms of data, nothing happens - including no changes to the user's data.

	"""

	FlowInitializing = enum.auto()
	FlowPending = enum.auto()
	NoFlow = enum.auto()

	@classmethod
	def all(cls) -> set[typ.Self]:
		"""Query all flow signals, using a simple cache to ensure minimal overhead when used in ex. `draw()` functions.

		Returns:
			Set of FlowSignal enum items, for easy `O(1)` lookup
		"""
		global _FLOW_SIGNAL_SET  # noqa: PLW0603

		if _FLOW_SIGNAL_SET is None:
			_FLOW_SIGNAL_SET = set(FlowSignal)

		return _FLOW_SIGNAL_SET

	@classmethod
	def check(cls, obj: typ.Any) -> set[typ.Self]:
		"""Checks whether an arbitrary object is a `FlowSignal` with tiny overhead.

		Notes:
			Optimized by first performing an `isinstance` check against both `FlowSignal` and `str`.
			Then, we can check membership in `cls.all()` with `O(1)`, since (by type narrowing like this) we've ensured that the object is hashable.

		Returns:
			Whether `obj` is a `FlowSignal`.

		Examples:
			A common pattern to ensure an object is **not** a `FlowSignal` is `not FlowSignal.check(obj)`.
		"""
		return isinstance(obj, FlowSignal | str) and obj in FlowSignal.all()

	@classmethod
	def check_single(cls, obj: typ.Any, single: typ.Self) -> set[typ.Self]:
		"""Checks whether an arbitrary object is a particular `FlowSignal`, with tiny overhead.

		Use this whenever it is important to make different decisions based on different `FlowSignal`s.

		Notes:
			Generally, you should use `cls.check()`.
			It tends to only be important to know whether you're getting a proper object from the flow, or whether it's dumping a `FlowSignal` on you instead.

			However, certain nodes might have a good reason to react differently .
			One example is deciding whether to keep node-internal caches around in the absence of data: `FlowSignal.FlowPending` hints to keep it around (allowing the user to ex. change selections, etc.), while `FlowSignal.NoFlow` hints to get rid of it immediately (resetting the node entirely).

		Returns:
			Whether `obj` is a `FlowSignal`.

		Examples:
			A common pattern to ensure an object is **not** a `FlowSignal` is `not FlowSignal.check(obj)`.
		"""
		return isinstance(obj, FlowSignal | str) and obj == single
