import dataclasses
import inspect
import typing as typ
from types import MappingProxyType

from ....utils import logger
from .. import contracts as ct

log = logger.get(__name__)

UnitSystemID = str
UnitSystem = dict[ct.SocketType, typ.Any]


####################
# - Event Callback Information
####################
@dataclasses.dataclass(kw_only=True, frozen=True)
class InfoDataChanged:
	run_on_init: bool
	on_changed_sockets: set[ct.SocketName]
	on_changed_props: set[str]
	on_any_changed_loose_input: set[str]


@dataclasses.dataclass(kw_only=True, frozen=True)
class InfoOutputRequested:
	output_socket_name: ct.SocketName
	any_loose_output_socket: bool
	kind: ct.DataFlowKind

	depon_props: set[str]

	depon_input_sockets: set[ct.SocketName]
	depon_input_socket_kinds: dict[ct.SocketName, ct.DataFlowKind]
	depon_all_loose_input_sockets: bool

	depon_output_sockets: set[ct.SocketName]
	depon_output_socket_kinds: dict[ct.SocketName, ct.DataFlowKind]
	depon_all_loose_output_sockets: bool


EventCallbackInfo: typ.TypeAlias = InfoDataChanged | InfoOutputRequested


####################
# - Event Decorator
####################
ManagedObjName: typ.TypeAlias = str
PropName: typ.TypeAlias = str


def event_decorator(
	action_type: ct.DataFlowAction,
	callback_info: EventCallbackInfo | None,
	stop_propagation: bool = False,
	# Request Data for Callback
	managed_objs: set[ManagedObjName] = frozenset(),
	props: set[PropName] = frozenset(),
	input_sockets: set[ct.SocketName] = frozenset(),
	input_sockets_optional: dict[ct.SocketName, bool] = MappingProxyType({}),
	input_socket_kinds: dict[ct.SocketName, ct.DataFlowKind] = MappingProxyType({}),
	output_sockets: set[ct.SocketName] = frozenset(),
	output_sockets_optional: dict[ct.SocketName, bool] = MappingProxyType({}),
	output_socket_kinds: dict[ct.SocketName, ct.DataFlowKind] = MappingProxyType({}),
	all_loose_input_sockets: bool = False,
	all_loose_output_sockets: bool = False,
	# Request Unit System Scaling
	unit_systems: dict[UnitSystemID, UnitSystem] = MappingProxyType({}),
	scale_input_sockets: dict[ct.SocketName, UnitSystemID] = MappingProxyType({}),
	scale_output_sockets: dict[ct.SocketName, UnitSystemID] = MappingProxyType({}),
):
	"""Returns a decorator for a method of `MaxwellSimNode`, declaring it as able respond to events passing through a node.

	Parameters:
		action_type: A name describing which event the decorator should respond to.
			Set to `return_method.action_type`
		callback_info: A dictionary that provides the caller with additional per-`action_type` information.
			This might include parameters to help select the most appropriate method(s) to respond to an event with, or actions to take after running the callback.
		props: Set of `props` to compute, then pass to the decorated method.
		stop_propagation: Whether or stop propagating the event through the graph after encountering this method.
			Other methods defined on the same node will still run.
		managed_objs: Set of `managed_objs` to retrieve, then pass to the decorated method.
		input_sockets: Set of `input_sockets` to compute, then pass to the decorated method.
		input_socket_kinds: The `ct.DataFlowKind` to compute per-input-socket.
			If an input socket isn't specified, it defaults to `ct.DataFlowKind.Value`.
		output_sockets: Set of `output_sockets` to compute, then pass to the decorated method.
		all_loose_input_sockets: Whether to compute all loose input sockets and pass them to the decorated method.
			Used when the names of the loose input sockets are unknown, but all of their values are needed.
		all_loose_output_sockets: Whether to compute all loose output sockets and pass them to the decorated method.
			Used when the names of the loose output sockets are unknown, but all of their values are needed.

	Returns:
		A decorator, which can be applied to a method of `MaxwellSimNode`.
		When a `MaxwellSimNode` subclass initializes, such a decorated method will be picked up on.

		When the `action_type` action passes through the node, then `callback_info` is used to determine
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
							kind=input_socket_kinds.get(
								input_socket_name, ct.DataFlowKind.Value
							),
							unit_system=(
								unit_system := unit_systems.get(
									scale_input_sockets.get(input_socket_name)
								)
							),
							optional=input_sockets_optional.get(
								input_socket_name, False
							),
						)
						for input_socket_name in input_sockets
					}
				}
				if input_sockets
				else {}
			)

			## Output Sockets
			method_kw_args |= (
				{
					'output_sockets': {
						output_socket_name: ct.DataFlowKind.scale_to_unit_system(
							(
								output_socket_kind := output_socket_kinds.get(
									output_socket_name, ct.DataFlowKind.Value
								)
							),
							node.compute_output(
								output_socket_name,
								kind=output_socket_kind,
								optional=output_sockets_optional.get(
									output_socket_name, False
								),
							),
							node.outputs[output_socket_name].socket_type,
							unit_systems.get(
								scale_output_sockets.get(output_socket_name)
							),
						)
						if scale_output_sockets.get(output_socket_name) is not None
						else node.compute_output(
							output_socket_name,
							kind=output_socket_kinds.get(
								output_socket_name, ct.DataFlowKind.Value
							),
							optional=output_sockets_optional.get(
								output_socket_name, False
							),
						)
						for output_socket_name in output_sockets
					}
				}
				if output_sockets
				else {}
			)

			# Loose Sockets
			## Compute All Loose Input Sockets
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

			# Call Method
			return method(
				node,
				**method_kw_args,
			)

		# Set Decorated Attributes and Return
		## Fix Introspection + Documentation
		# decorated.__name__ = method.__name__
		# decorated.__module__ = method.__module__
		# decorated.__qualname__ = method.__qualname__
		# decorated.__doc__ = method.__doc__

		## Add Spice
		decorated.action_type = action_type
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
		action_type=ct.DataFlowAction.EnableLock,
		callback_info=None,
		**kwargs,
	)


def on_disable_lock(
	**kwargs,
):
	return event_decorator(
		action_type=ct.DataFlowAction.DisableLock,
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
		action_type=ct.DataFlowAction.DataChanged,
		callback_info=InfoDataChanged(
			run_on_init=run_on_init,
			on_changed_sockets=(
				socket_name if isinstance(socket_name, set) else {socket_name}
			),
			on_changed_props=(prop_name if isinstance(prop_name, set) else {prop_name}),
			on_any_changed_loose_input=any_loose_input_socket,
		),
		**kwargs,
	)


## TODO: Change name to 'on_output_requested'
def computes_output_socket(
	output_socket_name: ct.SocketName | None,
	any_loose_output_socket: bool = False,
	kind: ct.DataFlowKind = ct.DataFlowKind.Value,
	**kwargs,
):
	return event_decorator(
		action_type=ct.DataFlowAction.OutputRequested,
		callback_info=InfoOutputRequested(
			output_socket_name=output_socket_name,
			any_loose_output_socket=any_loose_output_socket,
			kind=kind,
			depon_props=kwargs.get('props', set()),
			depon_input_sockets=kwargs.get('input_sockets', set()),
			depon_input_socket_kinds=kwargs.get('input_socket_kinds', set()),
			depon_output_sockets=kwargs.get('output_sockets', set()),
			depon_output_socket_kinds=kwargs.get('output_socket_kinds', set()),
			depon_all_loose_input_sockets=kwargs.get('all_loose_input_sockets', set()),
			depon_all_loose_output_sockets=kwargs.get(
				'all_loose_output_sockets', set()
			),
		),
		**kwargs,  ## stop_propagation has no effect.
	)


def on_show_preview(
	**kwargs,
):
	return event_decorator(
		action_type=ct.DataFlowAction.ShowPreview,
		callback_info={},
		**kwargs,
	)


def on_show_plot(
	stop_propagation: bool = True,
	**kwargs,
):
	return event_decorator(
		action_type=ct.DataFlowAction.ShowPlot,
		callback_info={},
		stop_propagation=stop_propagation,
		**kwargs,
	)
