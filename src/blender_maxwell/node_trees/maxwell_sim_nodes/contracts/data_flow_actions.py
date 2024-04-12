import enum
import typing as typ

import typing_extensions as typx


class DataFlowAction(enum.StrEnum):
	# Locking
	EnableLock = 'enable_lock'
	DisableLock = 'disable_lock'

	# Value
	OutputRequested = 'output_requested'
	DataChanged = 'value_changed'

	# Previewing
	ShowPreview = 'show_preview'
	ShowPlot = 'show_plot'

	def trigger_direction(action: typ.Self) -> typx.Literal['input', 'output']:
		"""When a given action is triggered, all sockets/nodes/... in this direction should be recursively triggered.

		Parameters:
			action: The action for which to retrieve the trigger direction.

		Returns:
			The trigger direction, which can be used ex. in nodes to select `node.inputs` or `node.outputs`.
		"""
		return {
			DataFlowAction.EnableLock: 'input',
			DataFlowAction.DisableLock: 'input',
			DataFlowAction.DataChanged: 'output',
			DataFlowAction.OutputRequested: 'input',
			DataFlowAction.ShowPreview: 'input',
			DataFlowAction.ShowPlot: 'input',
		}[action]

	def stop_if_no_event_methods(action: typ.Self) -> bool:
		return {
			DataFlowAction.EnableLock: False,
			DataFlowAction.DisableLock: False,
			DataFlowAction.DataChanged: True,
			DataFlowAction.OutputRequested: True,
			DataFlowAction.ShowPreview: False,
			DataFlowAction.ShowPlot: False,
		}[action]
