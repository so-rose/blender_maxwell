import enum
import inspect
import typing as typ
from types import MappingProxyType

from ....utils import extra_sympy_units as spux
from ....utils import logger
from .. import contracts as ct
from .base import MaxwellSimNode

log = logger.get(__name__)

UnitSystemID = str
UnitSystem = dict[ct.SocketType, typ.Any]


class EventCallbackType(enum.StrEnum):
	"""Names of actions that support callbacks."""

	computes_output_socket = enum.auto()
	on_value_changed = enum.auto()
	on_show_plot = enum.auto()
	on_init = enum.auto()


####################
# - Event Callback Information
####################
class EventCallbackData_ComputesOutputSocket(typ.TypedDict):  # noqa: N801
	"""Extra data used to select a method to compute output sockets."""

	output_socket_name: ct.SocketName
	kind: ct.DataFlowKind


class EventCallbackData_OnValueChanged(typ.TypedDict):  # noqa: N801
	"""Extra data used to select a method to compute output sockets."""

	changed_sockets: set[ct.SocketName]
	changed_props: set[str]
	changed_loose_input: set[str]


class EventCallbackData_OnShowPlot(typ.TypedDict):  # noqa: N801
	"""Extra data in the callback, used when showing a plot."""

	stop_propagation: bool


class EventCallbackData_OnInit(typ.TypedDict):  # noqa: D101, N801
	pass


EventCallbackData: typ.TypeAlias = (
	EventCallbackData_ComputesOutputSocket
	| EventCallbackData_OnValueChanged
	| EventCallbackData_OnShowPlot
	| EventCallbackData_OnInit
)


####################
# - Event Decorator
####################
ManagedObjName: typ.TypeAlias = str
PropName: typ.TypeAlias = str


def event_decorator(
	action_type: EventCallbackType,
	extra_data: EventCallbackData,
	kind: ct.DataFlowKind = ct.DataFlowKind.Value,
	props: set[PropName] = frozenset(),
	managed_objs: set[ManagedObjName] = frozenset(),
	input_sockets: set[ct.SocketName] = frozenset(),
	output_sockets: set[ct.SocketName] = frozenset(),
	all_loose_input_sockets: bool = False,
	all_loose_output_sockets: bool = False,
	unit_systems: dict[UnitSystemID, UnitSystem] = MappingProxyType({}),
	scale_input_sockets: dict[ct.SocketName, UnitSystemID] = MappingProxyType({}),
	scale_output_sockets: dict[ct.SocketName, UnitSystemID] = MappingProxyType({}),
):
	"""Returns a decorator for a method of `MaxwellSimNode`, declaring it as able respond to events passing through a node.

	Parameters:
		action_type: A name describing which event the decorator should respond to.
			Set to `return_method.action_type`
		extra_data: A dictionary that provides the caller with additional per-`action_type` information.
			This might include parameters to help select the most appropriate method(s) to respond to an event with, or actions to take after running the callback.
		kind: The `ct.DataFlowKind` used to compute all input and output socket data for methods with.
			Only affects data passed to the decorated method; namely `input_sockets`, `output_sockets`, and their loose variants.
		props: Set of `props` to compute, then pass to the decorated method.
		managed_objs: Set of `managed_objs` to retrieve, then pass to the decorated method.
		input_sockets: Set of `input_sockets` to compute, then pass to the decorated method.
		output_sockets: Set of `output_sockets` to compute, then pass to the decorated method.
		all_loose_input_sockets: Whether to compute all loose input sockets and pass them to the decorated method.
			Used when the names of the loose input sockets are unknown, but all of their values are needed.
		all_loose_output_sockets: Whether to compute all loose output sockets and pass them to the decorated method.
			Used when the names of the loose output sockets are unknown, but all of their values are needed.

	Returns:
		A decorator, which can be applied to a method of `MaxwellSimNode`.
		When a `MaxwellSimNode` subclass initializes, such a decorated method will be picked up on.

		When the `action_type` action passes through the node, then `extra_data` is used to determine
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

		# TODO: Check Function Annotation Validity
		## - socket capabilities

		def decorated(node: MaxwellSimNode):
			method_kw_args = {}  ## Keyword Arguments for Decorated Method

			# Compute Requested Props
			if props:
				_props = {prop_name: getattr(node, prop_name) for prop_name in props}
				method_kw_args |= {'props': _props}

			# Retrieve Requested Managed Objects
			if managed_objs:
				_managed_objs = {
					managed_obj_name: node.managed_objs[managed_obj_name]
					for managed_obj_name in managed_objs
				}
				method_kw_args |= {'managed_objs': _managed_objs}

			# Requested Sockets
			## Compute Requested Input Sockets
			if input_sockets:
				_input_sockets = {
					input_socket_name: node._compute_input(input_socket_name, kind)
					for input_socket_name in input_sockets
				}

				# Scale Specified Input Sockets to Unit System
				## First, scale the input socket value to the given unit system
				## Then, convert the symbol-less sympy scalar to a python type.
				for input_socket_name, unit_system_id in scale_input_sockets.items():
					unit_system = unit_systems[unit_system_id]
					_input_sockets[input_socket_name] = spux.sympy_to_python(
						spux.scale_to_unit(
							_input_sockets[input_socket_name],
							unit_system[node.inputs[input_socket_name].socket_type],
						)
					)

				method_kw_args |= {'input_sockets': _input_sockets}

			## Compute Requested Output Sockets
			if output_sockets:
				_output_sockets = {
					output_socket_name: node.compute_output(output_socket_name, kind)
					for output_socket_name in output_sockets
				}

				# Scale Specified Output Sockets to Unit System
				## First, scale the output socket value to the given unit system
				## Then, convert the symbol-less sympy scalar to a python type.
				for output_socket_name, unit_system_id in scale_output_sockets.items():
					unit_system = unit_systems[unit_system_id]
					_output_sockets[output_socket_name] = spux.sympy_to_python(
						spux.scale_to_unit(
							_output_sockets[output_socket_name],
							unit_system[node.outputs[output_socket_name].socket_type],
						)
					)
				method_kw_args |= {'output_sockets': _output_sockets}

			# Loose Sockets
			## Compute All Loose Input Sockets
			if all_loose_input_sockets:
				_loose_input_sockets = {
					input_socket_name: node._compute_input(input_socket_name, kind)
					for input_socket_name in node.loose_input_sockets
				}
				method_kw_args |= {'loose_input_sockets': _loose_input_sockets}

			## Compute All Loose Output Sockets
			if all_loose_output_sockets:
				_loose_output_sockets = {
					output_socket_name: node.compute_output(output_socket_name, kind)
					for output_socket_name in node.loose_output_sockets
				}
				method_kw_args |= {'loose_output_sockets': _loose_output_sockets}

			# Unit Systems
			if unit_systems:
				method_kw_args |= {'unit_systems': unit_systems}

			# Call Method
			return method(
				node,
				**method_kw_args,
			)

		# Set Decorated Attributes and Return
		## Fix Introspection + Documentation
		#decorated.__name__ = method.__name__
		#decorated.__module__ = method.__module__
		#decorated.__qualname__ = method.__qualname__
		#decorated.__doc__ = method.__doc__

		## Add Spice
		decorated.action_type = action_type
		decorated.extra_data = extra_data

		return decorated

	return decorator


####################
# - Simplified Event Callbacks
####################
def computes_output_socket(
	output_socket_name: ct.SocketName,
	kind: ct.DataFlowKind = ct.DataFlowKind.Value,
	**kwargs,
):
	return event_decorator(
		action_type='computes_output_socket',
		extra_data={
			'output_socket_name': output_socket_name,
			'kind': kind,
		},
		**kwargs,
	)


## TODO: Consider changing socket_name and prop_name to more obvious names.
def on_value_changed(
	socket_name: set[ct.SocketName] | ct.SocketName | None = None,
	prop_name: set[str] | str | None = None,
	any_loose_input_socket: bool = False,
	**kwargs,
):
	return event_decorator(
		action_type=EventCallbackType.on_value_changed,
		extra_data={
			'changed_sockets': (
				socket_name if isinstance(socket_name, set) else {socket_name}
			),
			'changed_props': (prop_name if isinstance(prop_name, set) else {prop_name}),
			'changed_loose_input': any_loose_input_socket,
		},
		**kwargs,
	)


def on_show_plot(
	stop_propagation: bool = False,
	**kwargs,
):
	return event_decorator(
		action_type=EventCallbackType.on_show_plot,
		extra_data={
			'stop_propagation': stop_propagation,
		},
		**kwargs,
	)


def on_init(**kwargs):
	return event_decorator(
		action_type=EventCallbackType.on_init,
		extra_data={},
		**kwargs,
	)
