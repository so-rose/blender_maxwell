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

import functools
import inspect
import typing as typ
import uuid
from collections import defaultdict
from fractions import Fraction
from types import MappingProxyType

import bpy
import jax
import numpy as np
import pydantic as pyd
import sympy as sp

from blender_maxwell.utils import logger
from blender_maxwell.utils import sympy_extra as spux
from blender_maxwell.utils.frozendict import FrozenDict, frozendict
from blender_maxwell.utils.lru_method import method_lru

from .. import contracts as ct

log = logger.get(__name__)

ManagedObjName: typ.TypeAlias = str
UnitSystemID = str

FK = ct.FlowKind
FS = ct.FlowSignal

EVENT_METHOD_IDENTIFIER: str = str(uuid.uuid4())  ## Changes on startup.


####################
# - Event Callback Information
####################
class CallbackInfo(pyd.BaseModel):
	"""Base for information associated with an event method callback."""

	model_config = pyd.ConfigDict(frozen=True)

	# def parse_for_method(self, node):


class InfoDataChanged(CallbackInfo):
	"""Information for determining whether a particular `DataChanged` callback should run."""

	model_config = pyd.ConfigDict(frozen=True)

	on_changed_props: frozenset[ct.PropName]
	on_changed_sockets_kinds: FrozenDict[ct.SocketName, frozenset[FK]]
	on_any_changed_loose_socket: bool
	run_on_init: bool

	optional_sockets: frozenset[ct.SocketName]
	stop_propagation: bool = False

	####################
	# - Computed Properties
	####################
	@functools.cached_property
	def on_changed_sockets(self) -> frozenset[ct.SocketName]:
		"""Input sockets with a `FlowKind` for which the method will run."""
		return frozenset(self.on_changed_sockets_kinds.keys())

	@functools.cached_property
	def optional_sockets_kinds(self) -> frozendict[ct.SocketName, frozenset[FK]]:
		"""Input `socket|kind`s for which the method can run, even when only a `FlowSignal` is available."""
		return {
			changed_socket: kinds
			for changed_socket, kinds in self.on_changed_sockets_kinds
			if changed_socket in self.optional_sockets
		}

	####################
	# - Methods
	####################
	@method_lru(maxsize=2048)
	def should_run(
		self,
		changed_props: frozenset[str] | None,
		changed_socket: ct.SocketName | None,
		changed_kinds: frozenset[FK] | None = None,
		socket_is_loose: bool = False,
	):
		"""Deduce whether this method should run in response to a particular set of changed inputs."""
		prop_triggered = changed_props is not None and any(
			changed_prop in self.on_changed_props for changed_prop in changed_props
		)

		socket_triggered = (
			changed_socket is not None
			and changed_kinds is not None
			and changed_socket in self.on_changed_sockets
			and any(
				changed_kind in self.on_changed_sockets_kinds[changed_socket]
				for changed_kind in changed_kinds
			)
		)

		loose_socket_triggered = (
			socket_is_loose
			and changed_socket is not None
			and self.on_any_changed_loose_socket
		)
		return socket_triggered or prop_triggered or loose_socket_triggered


class InfoOutputRequested(CallbackInfo):
	"""Information for determining which output socket method should run."""

	model_config = pyd.ConfigDict(frozen=True)

	output_socket_name: ct.SocketName
	kind: FK

	depon_props: frozenset[str]
	depon_input_sockets_kinds: FrozenDict[ct.SocketName, FK | frozenset[FK]]
	depon_output_sockets_kinds: FrozenDict[ct.SocketName, FK | frozenset[FK]]
	depon_all_loose_input_sockets: bool
	depon_all_loose_output_sockets: bool

	####################
	# - Computed Properties
	####################
	@functools.cached_property
	def depon_input_sockets(self) -> frozenset[ct.SocketName]:
		"""The input sockets depended on by this output socket method."""
		return frozenset(self.depon_input_sockets_kinds.keys())

	@functools.cached_property
	def depon_output_sockets(self) -> frozenset[ct.SocketName]:
		"""The output sockets depended on by this output socket method."""
		return frozenset(self.depon_output_sockets_kinds.keys())

	####################
	# - Methods
	####################
	@method_lru(maxsize=2048)
	def should_run(
		self,
		requested_socket: ct.SocketName,
		requested_kind: FK,
	):
		"""Deduce whether this method can compute the requested socket and kind."""
		return (
			requested_kind is self.kind and requested_socket == self.output_socket_name
		)

	@method_lru(maxsize=2048)
	def should_recompute(
		self,
		changed_props: frozenset[str] | None,
		changed_socket: ct.SocketName | None,
		changed_kinds: frozenset[FK] | None = None,
		socket_is_loose: bool = False,
	):
		"""Deduce whether this method needs to be recomputed after a change in a particular set of changed inputs."""
		prop_altered = changed_props is not None and any(
			changed_prop in self.depon_props for changed_prop in changed_props
		)

		socket_altered = (
			changed_socket is not None
			and changed_kinds is not None
			and changed_socket in self.depon_input_sockets
			and any(
				kind in self.depon_input_sockets_kinds[changed_socket]
				for kind in changed_kinds
			)
		)

		loose_socket_altered = (
			socket_is_loose
			and changed_socket is not None
			and self.depon_all_loose_input_sockets
		)

		return prop_altered or socket_altered or loose_socket_altered


EventCallbackInfo: typ.TypeAlias = InfoDataChanged | InfoOutputRequested


####################
# - Node Parsers
####################
def parse_node_mobjs(node: bpy.types.Node, mobjs: frozenset[ManagedObjName]) -> typ.Any:
	"""Retrieve the given managed objects."""
	return {mobj_name: node.managed_objs[mobj_name] for mobj_name in mobjs}


def parse_node_props(
	node: bpy.types.Node,
	props: frozenset[ct.PropName],
	prop_unit_systems: frozendict[ct.PropName, spux.UnitSystem | None],
) -> typ.Any:
	"""Compute the values of the given property names, w/optional scaling to a unit system.

	Raises:
		ValueError: If a unit system is specified for the property value, but the property value is not a `sympy` type.
	"""
	return frozendict(
		{
			prop: node.compute_prop(prop, unit_system=prop_unit_systems.get(prop))
			for prop in props
		}
	)


def parse_node_sck(
	node: bpy.types.Node,
	direc: typ.Literal['input', 'output'],
	sck: ct.SocketName,
	_kind: FK | None,
	unit_system: spux.UnitSystem | None,
) -> typ.Any:
	"""Compute a single value for `sck|kind|unit_system`.

	Parameters:
		node: The node to parse a socket value from.
		direc: Whether the socket to parse is an input or output socket.
		sck: The name of the socket to parse.
		_kind: The `FlowKind` of the socket to parse.
			When `None`, use the `.active_kind` attribute of the socket to determine what to parse.
		unit_system: The unit system with which to scale the socket value.

	Returns:
		The value of the socket over the given `FlowKind` lane, potentially scaled to a unitless scalar (if requested) in a manner specific to the `FlowKind`.
	"""
	# Deduce Kind
	## -> _kind=None denotes "use active_kind of socket".
	kind = node._bl_sockets(direc=direc)[sck].active_kind if _kind is None else _kind  # noqa: SLF001

	# Compute Socket Value
	if direc == 'input':
		return node._compute_input(sck, kind=kind, unit_system=unit_system)  # noqa: SLF001
	if direc == 'output':
		return node.compute_output(sck, kind=kind, unit_system=unit_system)

	raise TypeError


def parse_node_scks_kinds(
	node: bpy.types.Node,
	direc: typ.Literal['input', 'output'],
	scks_kinds: frozendict[ct.SocketName, frozenset[FK] | FK | None],
	scks_unit_system: spux.UnitSystem
	| frozendict[ct.SocketName, spux.UnitSystem | None],
) -> (
	frozendict[ct.SocketName, typ.Any]
	| frozendict[ct.SocketName, frozenset[typ.Any]]
	| None
):
	"""Retrieve the values for given input sockets and kinds, w/optional scaling to a unit system.

	In general, unless the socket name is specified in `scks_optional`, then whenever `FlowSignal` is encountered while computing sockets, the function will return `None` immediately.
	This process is "short-circuit", which is a partial optimization causing an immediate return before any other computing any other sockets.
	"""
	# Compute Socket Values
	## -> Every time we run `compute()`, we might encounter a FlowSignal.
	## -> If so, and the socket is 'optional', we let it be.
	## -> If not, and the socket is not 'optional', we return immediately.
	computed_scks = {}
	for sck, kinds in scks_kinds.items():
		# Extract Unit System
		if isinstance(scks_unit_system, dict | frozendict):
			unit_system = scks_unit_system.get(sck)
		else:
			unit_system = scks_unit_system

		flow_values = {}
		for kind in kinds if isinstance(kinds, set | frozenset) else {kinds}:
			flow = parse_node_sck(node, direc, sck, kind, unit_system)
			flow_values[kind] = flow

		if len(flow_values) == 1:
			computed_scks[sck] = next(iter(flow_values.values()))
		else:
			computed_scks[sck] = frozendict(flow_values)

	return frozendict(computed_scks)


####################
# - Utilities
####################
def freeze_pytree(
	pytree: None
	| str
	| int
	| Fraction
	| float
	| complex
	| set
	| frozenset
	| list
	| tuple
	| dict
	| frozendict,
):
	"""Conform an arbitrarily nested containers into their immutable equivalent."""
	if isinstance(pytree, set | frozenset):
		return frozenset(freeze_pytree(el) for el in pytree)
	if isinstance(pytree, list | tuple):
		return tuple(freeze_pytree(el) for el in pytree)
	if isinstance(pytree, np.ndarray | jax.Array):
		return tuple(freeze_pytree(el) for el in pytree.tolist())
	if isinstance(pytree, dict | frozendict):
		return frozendict({k: freeze_pytree(v) for k, v in pytree.items()})

	if isinstance(pytree, None | str | int | Fraction | float | complex):
		return pytree

	raise TypeError


def realize_known(
	sck: frozendict[typ.Literal[FK.Func, FK.Params], ct.FuncFlow | ct.ParamsFlow],
	freeze: bool = False,
	conformed: bool = False,
) -> int | float | tuple[int | float] | None:
	"""Realize a concrete preview-value from a `FuncFlow` and a `ParamsFlow`, when there are no unrealized symbols in `ParamsFlow`.

	It often happens that we absolutely need an _unrealized_ value for a node to work, eg. when producing a `Value` from the middle of a fully-lazy chain.
	Several complications arise when doing so, not least of which is how to handle the case where there are still-unrealized symbols.

	This method encapsulates all of these complexities into a single call, whose availability can be handled with a simple `None`-check.

	Parameters:
		sck: Mapping from dictionaries

	Examples:
		Within an event method depending on a socket `Socket: {FK.Func, FK.Params, ...}`, a realized value can
			Generally accessed by calling `event.realize_known

	Returns `None` when there are unrealized symbols, or either `Func` or `Params` is a `FlowSignal`.
	"""
	has_func = not FS.check(sck[FK.Func])
	has_params = not FS.check(sck[FK.Params])

	if has_func and has_params:
		realized = sck[FK.Func].realize(sck[FK.Params])
		if freeze:
			return freeze_pytree(realized)
		if conformed:
			func_output = sck[FK.Func].func_output
			if func_output is not None:
				return func_output.conform(realized)
			return realized
		return realized
	return None


def realize_preview(
	sck: frozendict[typ.Literal[FK.Func, FK.Params], ct.FuncFlow | ct.ParamsFlow],
) -> int | float | tuple[int | float]:
	"""Realize a concrete preview-value from a `FuncFlow` and a `ParamsFlow`.

	This particular operation is widely used in `on_value_changed` methods that update previews, since they must intercept both the function and parameter flows in order to respect ex. partially relized symbols and units.
	Usually, when such a thing happens, we support it in `event_decorator`.
	But in this case, the designs required were not resonating quite right - adding either too much specific complexity to input/output/loose fields, requiring too much "magic", etc. .

	Why not just ask users to intercept intercept the `Value` output?
	Several reasons.
	Firstly, it alone _absolutely cannot_ handle unrealized symbols, which while reasonable for a structure that should be **fully realized**, is not quite the desired functionality with preview-oriented workflows: In particular, we want we want to use the the stand-in `SimSymbol.preview_value_phy`, even though it's not always "accurate".
	Secondly, constructing the full `Value` output is slow, and introduces superfluous preview-dependencies.

	In this situation, the best balance is to provide this utility function.
	The user needs to deconstruct the eg. `input_sockets` parameter _anyway_ at least once; with this function, what would otherwise be an unweildy piece of realization logic is cleanly encapsulated.
	"""
	return sck[FK.Func].realize_preview(sck[FK.Params])


def mk_sockets_kinds(
	sockets: dict[ct.SocketName, frozenset[FK]]
	| dict[ct.SocketName, FK]
	| ct.SocketName
	| None,
	default_kinds: frozenset[FK] = frozenset(FK),
) -> frozendict[ct.SocketName, frozenset[FK]]:
	"""Normalize the given parameters to a standardized type."""
	# Deduce Triggered Socket -> SocketKinds
	## -> Normalize all valid inputs to frozendict[SocketName, set[FlowKind]].
	if sockets is None:
		return {}
	if isinstance(sockets, dict | frozendict):
		sockets_kinds = {
			socket: (
				kinds if isinstance(kinds := _kinds, set | frozenset) else {_kinds}
			)
			for socket, _kinds in sockets.items()
		}
	else:
		sockets_kinds = {
			socket: default_kinds
			for socket in (
				sockets if isinstance(sockets, set | frozenset) else [sockets]
			)
		}

	return frozendict(sockets_kinds)


####################
# - General Event Callbacks
####################
def event_decorator(  # noqa: C901, PLR0913, PLR0915
	event: ct.FlowEvent,
	callback_info: EventCallbackInfo | None,
	# Loading: Internal Data
	managed_objs: frozenset[ManagedObjName] = frozenset(),
	props: frozenset[ct.PropName] = frozenset(),
	# Loading: Input Sockets
	input_sockets: frozenset[ct.SocketName] = frozenset(),
	input_socket_kinds: frozendict[ct.SocketName, FK | frozenset[FK]] = frozendict(),
	inscks_kinds: frozendict[ct.SocketName, FK | frozenset[FK]] | None = None,
	input_sockets_optional: frozenset[ct.SocketName] = frozendict(),
	# Loading: Output Sockets
	output_sockets: frozenset[ct.SocketName] = frozenset(),
	output_socket_kinds: frozendict[ct.SocketName, FK | frozenset[FK]] = frozendict(),
	outscks_kinds: frozendict[ct.SocketName, FK | frozenset[FK]] | None = None,
	output_sockets_optional: frozenset[ct.SocketName] = frozenset(),
	# Loading: Loose Sockets
	all_loose_input_sockets: bool = False,
	loose_input_sockets_kind: frozenset[FK] | FK | None = None,
	all_loose_output_sockets: bool = False,
	loose_output_sockets_kind: frozenset[FK] | FK | None = None,
	# Loading: Unit System Scaling
	scale_props: frozendict[ct.PropName, spux.UnitSystem] = frozendict(),
	scale_input_sockets: frozendict[ct.SocketName, spux.UnitSystem] | None = None,
	scale_output_sockets: frozendict[ct.SocketName, spux.UnitSystem] | None = None,
	scale_loose_input_sockets: spux.UnitSystem | None = None,
	scale_loose_output_sockets: spux.UnitSystem | None = None,
):
	"""Low-level decorator declaring a special "event method" of `MaxwellSimNode`, which is able to handle `ct.FlowEvent`s passing through.

	Should generally be used via a high-level decorator such as `on_value_changed`.

	For more about how event methods are actually registered and run, please refer to the documentation of `MaxwellSimNode`.

	Parameters:
		event: A name describing which event the decorator should respond to.
		callback_info: A dictionary that provides the caller with additional per-`event` information.
			This might include parameters to help select the most appropriate method(s) to respond to an event with, or events to take after running the callback.
		managed_objs: Set of `managed_objs` to retrieve, then pass to the decorated method.
		props: Set of `props` to compute, then pass to the decorated method.
		input_sockets: Set of `input_sockets` to compute, then pass to the decorated method.
		input_sockets_optional: Allow the method will run even if one of these input socket values are a `ct.FlowSignal`.
		input_socket_kinds: The `FK` to compute per-input-socket.
			If an input socket isn't specified, it defaults to `FK.Value`.
		output_sockets: Set of `output_sockets` to compute, then pass to the decorated method.
		output_sockets_optional: Allow the method will run even if one of these output socket values are a `ct.FlowSignal`.
			When True, lack of socket will produce `ct.FlowSignal.NoFlow`, instead of throwing an error.
		output_socket_kinds: The `FK` to compute per-output-socket.
			If an output socket isn't specified, it defaults to `FK.Value`.
		all_loose_input_sockets: Whether to compute all loose input sockets and pass them to the decorated method.
			Used when the names of the loose input sockets are unknown, but all of their values are needed.
		all_loose_output_sockets: Whether to compute all loose output sockets and pass them to the decorated method.
			Used when the names of the loose output sockets are unknown, but all of their values are needed.
		scale_props: A mapping of input sockets to unit system string idenfiers, which causes the output of that input socket to be scaled to the given unit system.
		scale_input_sockets: A mapping of input sockets to unit system string idenfiers, which causes the output of that input socket to be scaled to the given unit system.
			This greatly simplifies the conformance of particular sockets to particular unit systems, when the socket value must be used in a unit-unaware manner.
		scale_output_sockets: A mapping of output sockets to unit system string idenfiers, which causes the output of that input socket to be scaled to the given unit system.
			This greatly simplifies the conformance of particular sockets to particular unit systems, when the socket value must be used in a unit-unaware manner.

	Returns:
		A decorator, which can be applied to a method of `MaxwellSimNode` to make it an "event method".
	"""
	req_params = (
		{'self'}
		| ({'props'} if props else frozenset())
		| ({'managed_objs'} if managed_objs else frozenset())
		| ({'input_sockets'} if input_sockets or inscks_kinds else frozenset())
		| ({'output_sockets'} if output_sockets or outscks_kinds else frozenset())
		| ({'loose_input_sockets'} if all_loose_input_sockets else frozenset())
		| ({'loose_output_sockets'} if all_loose_output_sockets else frozenset())
	)

	# Simplify I/O Under Naming
	if inscks_kinds is None:
		inscks_kinds = frozendict(
			{
				socket: input_socket_kinds.get(socket, FK.Value)
				for socket in input_sockets
			}
		)
	inscks_unit_system = (
		frozendict(scale_input_sockets) if scale_input_sockets is not None else None
	)
	inscks_optional = frozenset(input_sockets_optional)

	if outscks_kinds is None:
		outscks_kinds = frozendict(
			{
				output_socket: output_socket_kinds.get(output_socket, FK.Value)
				for output_socket in output_sockets
			}
		)
	outscks_unit_system = (
		frozendict(scale_output_sockets) if scale_output_sockets is not None else None
	)
	outscks_optional = frozenset(output_sockets_optional)

	## TODO: More ex. introspective checks and such, to make it really hard to write invalid methods.
	# TODO: Check Function Annotation Validity
	## - socket capabilities

	def decorator(method: typ.Callable) -> typ.Callable:  # noqa: C901
		# Check Function Signature Validity
		func_sig = frozenset(inspect.signature(method).parameters.keys())

		## -> Too Few Arguments
		if func_sig != req_params and func_sig.issubset(req_params):
			msg = f'Decorated method {method.__name__} is missing arguments {req_params - func_sig}'

		## -> Too Many Arguments
		if func_sig != req_params and func_sig.issuperset(req_params):
			msg = f'Decorated method {method.__name__} has superfluous arguments {func_sig - req_params}'
			raise ValueError(msg)

		def decorated(node: bpy.types.Node):  # noqa: C901, PLR0912
			method_kwargs = defaultdict(dict)

			# Managed Objects
			if managed_objs:
				method_kwargs['managed_objs'] = {}
				for mobj_name, mobj in parse_node_mobjs(node, managed_objs).items():
					method_kwargs['managed_objs'] |= {mobj_name: mobj}

			# Properties
			if props:
				method_kwargs['props'] = {}
				for prop, value in parse_node_props(node, props, scale_props).items():
					method_kwargs['props'] |= {prop: value}

			# Sockets
			if inscks_kinds:
				method_kwargs['input_sockets'] = {}
				for insck, flow in parse_node_scks_kinds(
					node,
					'input',
					inscks_kinds,
					inscks_unit_system,
				).items():
					has_flow = not FS.check(flow)
					if has_flow or insck in inscks_optional:
						method_kwargs['input_sockets'] |= {insck: flow}
					else:
						flow_signal = flow
						return flow_signal  # noqa: RET504

			if outscks_kinds:
				method_kwargs['output_sockets'] = {}
				for outsck, flow in parse_node_scks_kinds(
					node,
					'output',
					outscks_kinds,
					outscks_unit_system,
				).items():
					has_flow = not FS.check(flow)
					if has_flow or outsck in outscks_optional:
						method_kwargs['output_sockets'] |= {outsck: flow}
					else:
						flow_signal = flow
						return flow_signal  # noqa: RET504

			# Loose Sockets
			if all_loose_input_sockets:
				method_kwargs['loose_input_sockets'] = {}
				loose_inscks = frozenset(node.loose_input_sockets.keys())
				loose_inscks_kinds = {
					loose_insck: loose_input_sockets_kind
					for loose_insck in loose_inscks
				}

				for loose_insck, flow in parse_node_scks_kinds(
					node,
					'input',
					loose_inscks_kinds,
					scale_loose_input_sockets,
				).items():
					method_kwargs['loose_input_sockets'] |= {loose_insck: flow}

			if all_loose_output_sockets:
				method_kwargs['loose_output_sockets'] = {}
				loose_outscks = frozenset(node.loose_output_sockets.keys())
				loose_outscks_kinds = {
					loose_outsck: loose_output_sockets_kind
					for loose_outsck in loose_outscks
				}

				for loose_outsck, flow in parse_node_scks_kinds(
					node,
					'input',
					loose_outscks_kinds,
					scale_loose_output_sockets,
				).items():
					method_kwargs['loose_output_sockets'] |= {loose_outsck: flow}

			# Call Method
			return method(
				node,
				**method_kwargs,
			)

		# Wrap Decorated Method Attributes
		## -> We can't just @wraps(), since the call signature changed.
		decorated.__name__ = method.__name__
		# decorated.__module__ = method.__module__
		# decorated.__qualname__ = method.__qualname__
		decorated.__doc__ = method.__doc__

		# Add Spice
		decorated.identifier = EVENT_METHOD_IDENTIFIER
		decorated.event = event
		decorated.callback_info = callback_info

		return decorated

	return decorator


####################
# - Specific Event Callbacks
####################
def on_enable_lock(
	**kwargs,
):
	"""Declare a method that reacts to the enabling of the interface lock."""
	return event_decorator(
		event=ct.FlowEvent.EnableLock,
		callback_info=None,
		**kwargs,
	)


def on_disable_lock(
	**kwargs,
):
	"""Declare a method that reacts to the disabling of the interface lock."""
	return event_decorator(
		event=ct.FlowEvent.DisableLock,
		callback_info=None,
		**kwargs,
	)


## TODO: Consider changing socket_name and prop_name to more obvious names.
def on_value_changed(
	prop_name: frozenset[str] | str | None = None,
	socket_name: frozendict[ct.SocketName, frozenset[FK]]
	| frozendict[ct.SocketName, FK]
	| frozenset[ct.SocketName]
	| ct.SocketName
	| None = None,
	socket_name_kinds: frozenset[FK] | FK = frozenset(FK),
	any_loose_input_socket: bool = False,
	run_on_init: bool = False,
	stop_propagation: bool = False,
	**kwargs,
):
	"""Declare a method that reacts to a change in node data.

	Can be configured to react to changes in:

	- Any particular input `socket|kind`s.
	- Any `loose_socket|active_kind`.
	- Any particular property.

	In addition, the method can be configured to run when its node is created and initialized for the first time.
	"""
	return event_decorator(
		event=ct.FlowEvent.DataChanged,
		callback_info=InfoDataChanged(
			# Trigger: Props
			on_changed_props=(
				prop_name
				if isinstance(prop_name, set | frozenset)
				else ({prop_name} if prop_name is not None else set())
			),
			# Trigger: Sockets
			on_changed_sockets_kinds=mk_sockets_kinds(socket_name, socket_name_kinds),
			# Trigger: Loose Sockets
			on_any_changed_loose_socket=any_loose_input_socket,
			# Trigger: Init
			run_on_init=run_on_init,
			# Hints
			optional_sockets={
				socket_name
				for socket_name in kwargs.get('input_sockets', set())
				if socket_name in kwargs.get('input_sockets_optional', set())
			},
			stop_propagation=stop_propagation,
		),
		**kwargs,
	)


def computes_output_socket(
	output_socket_name: ct.SocketName | None,
	kind: FK = FK.Value,
	**kwargs,
):
	"""Declare a method used to compute the value of a particular output `socket|kind`.

	The method's dependencies on properties, input/output `socket|kind`s, and loose input/output `socket|active_kind`s are recorded in its `callback_info`, such that the associated output socket cache can be appropriately invalidated whenever an input dependency changes.
	"""
	input_socket_kinds = kwargs.get('input_socket_kinds', {})
	depon_inscks_kinds = {
		insck: input_socket_kinds.get(insck, FK.Value)
		for insck in kwargs.get('input_sockets', set())
	} | kwargs.get('inscks_kinds', {})

	output_sockets_kinds = kwargs.get('output_sockets_kinds', {})
	depon_outscks_kinds = {
		outsck: output_sockets_kinds.get(outsck, FK.Value)
		for outsck in kwargs.get('output_sockets', set())
	} | kwargs.get('outscks_kinds', {})

	log.debug(
		[
			output_socket_name,
			kind,
			kwargs.get('props', set()),
			mk_sockets_kinds(depon_inscks_kinds),
			mk_sockets_kinds(depon_outscks_kinds),
			kwargs.get('all_loose_input_sockets', False),
			kwargs.get('all_loose_output_sockets', False),
		]
	)
	return event_decorator(
		event=ct.FlowEvent.OutputRequested,
		callback_info=InfoOutputRequested(
			output_socket_name=output_socket_name,
			kind=kind,
			# Dependency: Props
			depon_props=kwargs.get('props', set()),
			# Dependency: Input Sockets
			depon_input_sockets_kinds=mk_sockets_kinds(depon_inscks_kinds),
			# Dependency: Output Sockets
			depon_output_sockets_kinds=mk_sockets_kinds(depon_outscks_kinds),
			# Dependency: Loose Sockets
			depon_all_loose_input_sockets=kwargs.get('all_loose_input_sockets', False),
			depon_all_loose_output_sockets=kwargs.get(
				'all_loose_output_sockets', False
			),
		),
		**kwargs,
	)


def on_show_plot(
	**kwargs,
):
	"""Declare a method that reacts to a request to show a plot."""
	return event_decorator(
		event=ct.FlowEvent.ShowPlot,
		callback_info=None,
		**kwargs,
	)
