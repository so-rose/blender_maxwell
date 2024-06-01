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
import inspect
import typing as typ
from types import MappingProxyType

from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils import logger

from .. import contracts as ct

log = logger.get(__name__)

UnitSystemID = str


####################
# - Event Callback Information
####################
@dataclasses.dataclass(kw_only=True, frozen=True)
class InfoDataChanged:
	run_on_init: bool
	on_changed_sockets: set[ct.SocketName]
	on_changed_props: set[str]
	on_any_changed_loose_input: set[str]
	must_load_sockets: set[str]


@dataclasses.dataclass(kw_only=True, frozen=True)
class InfoOutputRequested:
	output_socket_name: ct.SocketName
	kind: ct.FlowKind

	depon_props: set[str]

	depon_input_sockets: set[ct.SocketName]
	depon_input_socket_kinds: dict[ct.SocketName, ct.FlowKind | set[ct.FlowKind]]
	depon_all_loose_input_sockets: bool

	depon_output_sockets: set[ct.SocketName]
	depon_output_socket_kinds: dict[ct.SocketName, ct.FlowKind | set[ct.FlowKind]]
	depon_all_loose_output_sockets: bool


EventCallbackInfo: typ.TypeAlias = InfoDataChanged | InfoOutputRequested


####################
# - Event Decorator
####################
ManagedObjName: typ.TypeAlias = str
PropName: typ.TypeAlias = str


def event_decorator(  # noqa: PLR0913
	event: ct.FlowEvent,
	callback_info: EventCallbackInfo | None,
	stop_propagation: bool = False,
	# Request Data for Callback
	managed_objs: set[ManagedObjName] = frozenset(),
	props: set[PropName] = frozenset(),
	input_sockets: set[ct.SocketName] = frozenset(),
	input_sockets_optional: dict[ct.SocketName, bool] = MappingProxyType({}),
	input_socket_kinds: dict[
		ct.SocketName, ct.FlowKind | set[ct.FlowKind]
	] = MappingProxyType({}),
	output_sockets: set[ct.SocketName] = frozenset(),
	output_sockets_optional: dict[ct.SocketName, bool] = MappingProxyType({}),
	output_socket_kinds: dict[
		ct.SocketName, ct.FlowKind | set[ct.FlowKind]
	] = MappingProxyType({}),
	all_loose_input_sockets: bool = False,
	all_loose_output_sockets: bool = False,
	# Request Unit System Scaling
	unit_systems: dict[UnitSystemID, spux.UnitSystem] = MappingProxyType({}),
	scale_input_sockets: dict[ct.SocketName, UnitSystemID] = MappingProxyType({}),
	scale_output_sockets: dict[ct.SocketName, UnitSystemID] = MappingProxyType({}),
):
	"""Low-level decorator declaring a special "event method" of `MaxwellSimNode`, which is able to handle `ct.FlowEvent`s passing through.

	Should generally be used via a high-level decorator such as `on_value_changed`.

	For more about how event methods are actually registered and run, please refer to the documentation of `MaxwellSimNode`.

	Parameters:
		event: A name describing which event the decorator should respond to.
		callback_info: A dictionary that provides the caller with additional per-`event` information.
			This might include parameters to help select the most appropriate method(s) to respond to an event with, or events to take after running the callback.
		stop_propagation: Whether or stop propagating the event through the graph after encountering this method.
			Other methods defined on the same node will still run.
		managed_objs: Set of `managed_objs` to retrieve, then pass to the decorated method.
		props: Set of `props` to compute, then pass to the decorated method.
		input_sockets: Set of `input_sockets` to compute, then pass to the decorated method.
		input_sockets_optional: Whether an input socket is required to exist.
			When True, lack of socket will produce `ct.FlowSignal.NoFlow`, instead of throwing an error.
		input_socket_kinds: The `ct.FlowKind` to compute per-input-socket.
			If an input socket isn't specified, it defaults to `ct.FlowKind.Value`.
		output_sockets: Set of `output_sockets` to compute, then pass to the decorated method.
		output_sockets_optional: Whether an output socket is required to exist.
			When True, lack of socket will produce `ct.FlowSignal.NoFlow`, instead of throwing an error.
		output_socket_kinds: The `ct.FlowKind` to compute per-output-socket.
			If an output socket isn't specified, it defaults to `ct.FlowKind.Value`.
		all_loose_input_sockets: Whether to compute all loose input sockets and pass them to the decorated method.
			Used when the names of the loose input sockets are unknown, but all of their values are needed.
		all_loose_output_sockets: Whether to compute all loose output sockets and pass them to the decorated method.
			Used when the names of the loose output sockets are unknown, but all of their values are needed.
		unit_systems: String identifiers under which to load a unit system, made available to the method.
		scale_input_sockets: A mapping of input sockets to unit system string idenfiers, which causes the output of that input socket to be scaled to the given unit system.
			This greatly simplifies the conformance of particular sockets to particular unit systems, when the socket value must be used in a unit-unaware manner.
		scale_output_sockets: A mapping of output sockets to unit system string idenfiers, which causes the output of that input socket to be scaled to the given unit system.
			This greatly simplifies the conformance of particular sockets to particular unit systems, when the socket value must be used in a unit-unaware manner.

	Returns:
		A decorator, which can be applied to a method of `MaxwellSimNode` to make it an "event method".
	"""
	req_params = (
		{'self'}
		| ({'props'} if props else set())
		| ({'managed_objs'} if managed_objs else set())
		| ({'input_sockets'} if input_sockets else set())
		| ({'output_sockets'} if output_sockets else set())
		| ({'loose_input_sockets'} if all_loose_input_sockets else set())
		| ({'loose_output_sockets'} if all_loose_output_sockets else set())
		| ({'unit_systems'} if unit_systems else set())
	)

	# TODO: Check that all Unit System IDs referenced are also defined in 'unit_systems'.
	## TODO: More ex. introspective checks and such, to make it really hard to write invalid methods.
	# TODO: Check Function Annotation Validity
	## - socket capabilities

	def decorator(method: typ.Callable) -> typ.Callable:
		# Check Function Signature Validity
		func_sig = set(inspect.signature(method).parameters.keys())

		## Too Few Arguments
		if func_sig != req_params and func_sig.issubset(req_params):
			msg = f'Decorated method {method.__name__} is missing arguments {req_params - func_sig}'

		## Too Many Arguments
		if func_sig != req_params and func_sig.issuperset(req_params):
			msg = f'Decorated method {method.__name__} has superfluous arguments {func_sig - req_params}'
			raise ValueError(msg)

		def decorated(node):
			method_kw_args = {}  ## Keyword Arguments for Decorated Method

			# Unit Systems
			method_kw_args |= {'unit_systems': unit_systems} if unit_systems else {}

			# Properties
			method_kw_args |= (
				{'props': {prop_name: getattr(node, prop_name) for prop_name in props}}
				if props
				else {}
			)

			# Managed Objects
			method_kw_args |= (
				{
					'managed_objs': {
						managed_obj_name: node.managed_objs[managed_obj_name]
						for managed_obj_name in managed_objs
					}
				}
				if managed_objs
				else {}
			)

			# Sockets
			## Input Sockets
			method_kw_args |= (
				{
					'input_sockets': {
						input_socket_name: node._compute_input(
							input_socket_name,
							kind=_kind,
							unit_system=(
								unit_systems.get(
									scale_input_sockets.get(input_socket_name)
								)
							),
							optional=input_sockets_optional.get(
								input_socket_name, False
							),
						)
						if not isinstance(
							_kind := input_socket_kinds.get(
								input_socket_name, ct.FlowKind.Value
							),
							set,
						)
						else {
							kind: node._compute_input(
								input_socket_name,
								kind=kind,
								unit_system=unit_systems.get(
									scale_input_sockets.get(input_socket_name)
								),
								optional=input_sockets_optional.get(
									input_socket_name, False
								),
							)
							for kind in _kind
						}
						for input_socket_name in input_sockets
					}
				}
				if input_sockets
				else {}
			)

			## Output Sockets
			def _g_output_socket(output_socket_name: ct.SocketName, kind: ct.FlowKind):
				if scale_output_sockets.get(output_socket_name) is None:
					return node.compute_output(
						output_socket_name,
						kind=kind,
						optional=output_sockets_optional.get(output_socket_name, False),
					)

				return ct.FlowKind.scale_to_unit_system(
					kind,
					node.compute_output(
						output_socket_name,
						kind=kind,
						optional=output_sockets_optional.get(output_socket_name, False),
					),
					unit_systems.get(scale_output_sockets.get(output_socket_name)),
				)

			method_kw_args |= (
				{
					'output_sockets': {
						output_socket_name: _g_output_socket(output_socket_name, _kind)
						if not isinstance(
							_kind := output_socket_kinds.get(
								output_socket_name, ct.FlowKind.Value
							),
							set,
						)
						else {
							kind: _g_output_socket(output_socket_name, kind)
							for kind in _kind
						}
						for output_socket_name in output_sockets
					}
				}
				if output_sockets
				else {}
			)

			# Loose Sockets
			## -> Determined by the active_kind of each loose input socket.
			method_kw_args |= (
				{
					'loose_input_sockets': {
						input_socket_name: node._compute_input(
							input_socket_name,
							kind=node.inputs[input_socket_name].active_kind,
						)
						for input_socket_name in node.loose_input_sockets
					}
				}
				if all_loose_input_sockets
				else {}
			)

			## Compute All Loose Output Sockets
			method_kw_args |= (
				{
					'loose_output_sockets': {
						output_socket_name: node.compute_output(
							output_socket_name,
							kind=node.outputs[output_socket_name].active_kind,
						)
						for output_socket_name in node.loose_output_sockets
					}
				}
				if all_loose_output_sockets
				else {}
			)

			# Propagate Initialization
			## If there is a FlowInitializing, then the method would fail.
			## Therefore, propagate FlowInitializing if found.
			if any(
				ct.FlowSignal.check_single(value, ct.FlowSignal.FlowInitializing)
				for sockets in [
					method_kw_args.get('input_sockets', {}),
					method_kw_args.get('loose_input_sockets', {}),
					method_kw_args.get('output_sockets', {}),
					method_kw_args.get('loose_output_sockets', {}),
				]
				for value in sockets.values()
			):
				return ct.FlowSignal.FlowInitializing

			# Call Method
			return method(
				node,
				**method_kw_args,
			)

		# Set Decorated Attributes and Return
		## TODO: Fix Introspection + Documentation
		# decorated.__name__ = method.__name__
		# decorated.__module__ = method.__module__
		# decorated.__qualname__ = method.__qualname__
		decorated.__doc__ = method.__doc__

		## Add Spice
		decorated.event = event
		decorated.callback_info = callback_info
		decorated.stop_propagation = stop_propagation

		return decorated

	return decorator


####################
# - Simplified Event Callbacks
####################
def on_enable_lock(
	**kwargs,
):
	return event_decorator(
		event=ct.FlowEvent.EnableLock,
		callback_info=None,
		**kwargs,
	)


def on_disable_lock(
	**kwargs,
):
	return event_decorator(
		event=ct.FlowEvent.DisableLock,
		callback_info=None,
		**kwargs,
	)


## TODO: Consider changing socket_name and prop_name to more obvious names.
def on_value_changed(
	socket_name: set[ct.SocketName] | ct.SocketName | None = None,
	prop_name: set[str] | str | None = None,
	any_loose_input_socket: bool = False,
	run_on_init: bool = False,
	**kwargs,
):
	return event_decorator(
		event=ct.FlowEvent.DataChanged,
		callback_info=InfoDataChanged(
			# Triggers
			run_on_init=run_on_init,
			on_changed_sockets=(
				socket_name if isinstance(socket_name, set) else {socket_name}
			),
			on_changed_props=(prop_name if isinstance(prop_name, set) else {prop_name}),
			on_any_changed_loose_input=any_loose_input_socket,
			# Loaded
			must_load_sockets={
				socket_name
				for socket_name in kwargs.get('input_sockets', {})
				if socket_name not in kwargs.get('input_sockets_optional', {})
			},
		),
		**kwargs,
	)


def computes_output_socket(
	output_socket_name: ct.SocketName | None,
	kind: ct.FlowKind = ct.FlowKind.Value,
	**kwargs,
):
	return event_decorator(
		event=ct.FlowEvent.OutputRequested,
		callback_info=InfoOutputRequested(
			output_socket_name=output_socket_name,
			kind=kind,
			depon_props=kwargs.get('props', set()),
			depon_input_sockets=kwargs.get('input_sockets', set()),
			depon_input_socket_kinds=kwargs.get('input_socket_kinds', {}),
			depon_output_sockets=kwargs.get('output_sockets', set()),
			depon_output_socket_kinds=kwargs.get('output_socket_kinds', {}),
			depon_all_loose_input_sockets=kwargs.get('all_loose_input_sockets', False),
			depon_all_loose_output_sockets=kwargs.get(
				'all_loose_output_sockets', False
			),
		),
		**kwargs,  ## stop_propagation has no effect.
	)


def on_show_preview(
	**kwargs,
):
	return event_decorator(
		event=ct.FlowEvent.ShowPreview,
		callback_info={},
		**kwargs,
	)


def on_show_plot(
	stop_propagation: bool = True,
	**kwargs,
):
	return event_decorator(
		event=ct.FlowEvent.ShowPlot,
		callback_info={},
		stop_propagation=stop_propagation,
		**kwargs,
	)
