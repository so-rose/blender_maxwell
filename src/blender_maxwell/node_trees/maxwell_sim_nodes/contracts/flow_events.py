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

from blender_maxwell.utils.staticproperty import staticproperty


class FlowEvent(enum.StrEnum):
	"""Defines an event that can propagate through the graph (node-socket-node-...).

	Contrary to `FlowKind`, a `FlowEvent` doesn't propagate any data.
	Instead, it allows for dead-simple communication across direct graph connections.

	The entire system is built around user-defined event handlers, which are also used internally.
	See `events`.

	Attributes:
		EnableLock: Indicates that the node/socket should enable locking.
			Locking prevents the use of the UI, including adding/removing links.
			This event can lock a subset of the node tree graph.
		DisableLock: Indicates that the node/socket should disable locking.
			This event can unlock part of a locked subgraph.
		LinkChanged: Indicates that a link to a node/socket was added/removed.
			Is translated to `DataChanged` on sockets before propagation.
		DataChanged: Indicates that data flowing through a node/socket was altered.
			In nodes, this event is accompanied by a `socket_name` or `prop_name`, to indicate which socket/property it is that was changed.
			**This event is essential**, as it invalidates all input/output socket caches along its path.
	"""

	# Lock Events
	EnableLock = enum.auto()
	DisableLock = enum.auto()

	# Data Events
	LinkChanged = enum.auto()
	DataChanged = enum.auto()

	# Non-Triggered Events
	ShowPlot = enum.auto()
	OutputRequested = enum.auto()

	# Properties
	@staticproperty
	def flow_direction() -> typ.Literal['input', 'output']:
		"""Describes the direction in which the event should flow.

		Doesn't include `FlowEvent`s that aren't meant to be triggered:
		- `OutputRequested`.

		Parameters:
			event: The event for which to retrieve the trigger direction.

		Returns:
			The trigger direction, which can be used ex. in nodes to select `node.inputs` or `node.outputs`.
		"""
		return {
			# Lock Events
			FlowEvent.EnableLock: 'input',
			FlowEvent.DisableLock: 'input',
			# Data Events
			FlowEvent.LinkChanged: 'output',
			FlowEvent.DataChanged: 'output',
		}
