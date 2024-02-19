import typing as typ
import typing_extensions as pytypes_ext

import bpy
import pydantic as pyd

from .. import contracts
from .. import sockets

####################
# - Decorator: Output Socket Computation
####################
@typ.runtime_checkable
class ComputeOutputSocketFunc(typ.Protocol[contracts.SocketReturnType]):
	"""Protocol describing a function that computes the value of an
	output socket.
	"""
	
	def __call__(
		_self,
		self: contracts.NodeTypeProtocol,
	) -> contracts.SocketReturnType:
		"""Describes the function signature of all functions that compute
		the value of an output socket.
		
		Args:
			node: A node in the tree, passed via the 'self' attribute of the
				node.
		
		Returns:
			The value of the output socket, as the relevant type.
		"""
		...

class PydanticProtocolMeta(type(pyd.BaseModel), type(typ.Protocol)): pass

class FuncOutputSocket(
	pyd.BaseModel,
	typ.Generic[contracts.SocketReturnType],
	ComputeOutputSocketFunc[contracts.SocketReturnType],
	metaclass=PydanticProtocolMeta,
):
	"""Defines a function (-like object) that defines an attachment from
	an output socket name, to the original method that computes the value of
	an output socket.
	
	Conforms to the protocol `ComputeOutputSocketFunc`.
	Validation is provided by subtyping `pydantic.BaseModel`.
	
	Attributes:
		output_socket_func: The original method computing the value of an
			output socket.
		output_socket_name: The SocketName of the output socket for which
			this function should be called to compute.
	"""
	
	output_socket_func: typ.Callable[
		[contracts.NodeTypeProtocol],
		contracts.SocketReturnType,
	]
	output_socket_name: contracts.SocketName
	
	def __call__(
		self,
		node: contracts.NodeTypeProtocol
	) -> contracts.SocketReturnType:
		"""Computes the value of an output socket.
		
		Args:
			node: A node in the tree, passed via the 'self' attribute of the
				node.
		
		Returns:
			The value of the output socket, as the relevant type.
		"""
		
		return self.output_socket_func(node)

# Define Factory Function & Decorator 
def computes_output_socket(
	output_socket_name: contracts.SocketName,
) -> typ.Callable[
	[ComputeOutputSocketFunc[contracts.SocketReturnType]],
	FuncOutputSocket[contracts.SocketReturnType],
]:
	"""Given a socket name, defines a function-that-makes-a-function (aka.
	decorator) which has the name of the socket attached.
	
	Must be used as a decorator, ex. `@compute_output_socket("name")`.
	
	Args:
		output_socket_name: The name of the output socket to attach the
			decorated method to.
	
	Returns:
		The decorator, which takes the output-socket-computing method
			and returns a new output-socket-computing method, now annotated
			and discoverable by the `MaxwellSimTreeNode`.
	"""
	
	def decorator(
		output_socket_func: ComputeOutputSocketFunc[contracts.SocketReturnType]
	) -> FuncOutputSocket[contracts.SocketReturnType]:
		return FuncOutputSocket(
			output_socket_func=output_socket_func,
			output_socket_name=output_socket_name,
		)
	
	return decorator



####################
# - Node Callbacks
####################
def sync_selected_preset(node) -> None:
	"""Whenever a preset is set in a NodeTypeProtocol, this function
	should be called to overwrite the `default_value`s of the input sockets
	with the actual preset values.
	
	Args:
		node: The node for which input socket `default_value`s should be
			set to the values defined within the currently selected preset.
	"""
	if hasattr(node, "preset") and hasattr(node, "presets"):
		if node.preset is None:
			msg = f"Node {node} has no preset EnumProperty"
			raise ValueError(msg)
	
		if node.presets is None:
			msg = f"Node {node} has preset EnumProperty, but no defined presets."
			raise ValueError(msg)
	
	# Set Input Sockets to Preset Values
	preset_def = node.presets[node.preset]
	for input_socket_name, value in preset_def.values.items():
		node.s_input_value(input_socket_name, value)



####################
# - Node Superclass Definition
####################
class MaxwellSimTreeNode(bpy.types.Node):
	"""A base type for nodes that greatly simplifies the implementation of
	reliable, powerful nodes.
	
	Should be used together with `contracts.NodeTypeProtocol`.
	"""
	def __init_subclass__(cls, **kwargs: typ.Any):
		super().__init_subclass__(**kwargs)  ## Yucky superclass setup.
		
		# Set bl_idname
		cls.bl_idname = cls.node_type.value 
		
		# Declare Node Property: 'preset' EnumProperty
		if hasattr(cls, "input_socket_sets") or hasattr(cls, "output_socket_sets"):
			if not hasattr(cls, "input_socket_sets"):
				cls.input_socket_sets = {}
			if not hasattr(cls, "output_socket_sets"):
				cls.output_socket_sets = {}
			
			socket_set_keys = [
				input_socket_set_key
				for input_socket_set_key in cls.input_socket_sets.keys()
			]
			socket_set_keys += [
				output_socket_set_key
				for output_socket_set_key in cls.output_socket_sets.keys()
				if output_socket_set_key not in socket_set_keys
			]
			
			cls.__annotations__["socket_set"] = bpy.props.EnumProperty(
				name="",
				description="Select a node socket configuration",
				items=[
					(
						socket_set_key,
						socket_set_key.capitalize(),
						socket_set_key.capitalize(),
					)
					for socket_set_key in socket_set_keys
				],
				default=socket_set_keys[0],
				update=(lambda self, context: self._update_socket()),
			)
			cls.__annotations__["socket_set_previous"] = bpy.props.StringProperty(
				default=socket_set_keys[0]
			)
		
		# Declare Node Property: 'preset' EnumProperty
		if hasattr(cls, "presets"):
			first_preset = list(cls.presets.keys())[0]
			cls.__annotations__["preset"] = bpy.props.EnumProperty(
				name="Presets",
				description="Select a preset",
				items=[
					(
						preset_name,
						preset_def.label,
						preset_def.description,
					)
					for preset_name, preset_def in cls.presets.items()
				],
				default=first_preset,  ## 1st is Default
				update=(lambda self, context: sync_selected_preset(self)),
			)
		else:
			cls.preset = None
			cls.presets = None
	
	####################
	# - Blender Init / Constraints
	####################
	def init(self, context: bpy.types.Context):
		"""Declares input and output sockets as described by the
		`NodeTypeProtocol` specification, and initializes each as described
		by user-provided `SocketDefProtocol`s.
		"""
		# Initialize Input Sockets
		for socket_name, socket_def in self.input_sockets.items():
			self.inputs.new(
				socket_def.socket_type.value,  ## strenum.value => a real str
				socket_def.label,
			)
			
			# Retrieve the Blender Socket (bpy.types.NodeSocket)
			## We could use self.g_input_bl_socket()...
			## ...but that would rely on implicit semi-initialized state.
			bl_socket = self.inputs[
				self.input_sockets[socket_name].label
			]
			
			# Initialize the Socket from the Socket Definition
			## `bl_socket` knows whether it's an input or output socket...
			## ...via its `.is_output` attribute.
			socket_def.init(bl_socket)
		
		# Initialize Output Sockets
		for socket_name, socket_def in self.output_sockets.items():
			self.outputs.new(
				socket_def.socket_type.value,
				socket_def.label,
			)
			
			bl_socket = self.outputs[
				self.output_sockets[socket_name].label
			]
			socket_def.init(bl_socket)
		
		# Initialize Dynamic Sockets
		if hasattr(self, "socket_set"):
			if self.socket_set in self.input_socket_sets:
				for socket_name, socket_def in self.input_socket_sets[self.socket_set].items():
					self.inputs.new(
						socket_def.socket_type.value,
						socket_def.label,
					)
					
					bl_socket = self.inputs[socket_def.label]
					socket_def.init(bl_socket)
			
			if self.socket_set in self.output_socket_sets:
				for socket_name, socket_def in self.output_socket_sets[self.socket_set].items():
					self.outputs.new(
						socket_def.socket_type.value,
						socket_def.label,
					)
					
					bl_socket = self.outputs[socket_def.label]
					socket_def.init(bl_socket)
		
		# Sync Default Preset to Input Socket Values
		if self.preset is not None:
			sync_selected_preset(self)
	
	@classmethod
	def poll(cls, ntree: bpy.types.NodeTree) -> bool:
		"""This class method controls whether a node can be instantiated
		in a given node tree.
		
		In our case, we restrict node instantiation to within a
		MaxwellSimTree.
		
		Args:
			ntree: The node tree within which the user is currently working.
		
		Returns:
			Whether or not the user should be able to instantiate the node.
		
		"""
		
		return ntree.bl_idname == contracts.TreeType.MaxwellSim.value
	
	def _update_socket(self):
		if not hasattr(self, "socket_set"):
			raise ValueError("no socket")
		
		if self.socket_set == self.socket_set_previous: return
		
		# Delete Old Sockets
		if self.socket_set_previous in self.input_socket_sets:
			for socket_name, socket_def in self.input_socket_sets[self.socket_set_previous].items():
				bl_socket = self.inputs[socket_def.label]
				self.inputs.remove(bl_socket)
		
		if self.socket_set_previous in self.output_socket_sets:
			for socket_name, socket_def in self.output_socket_sets[self.socket_set_previous].items():
				bl_socket = self.outputs[socket_def.label]
				self.outputs.remove(bl_socket)
		
		# Add New Sockets
		if self.socket_set in self.input_socket_sets:
			for socket_name, socket_def in self.input_socket_sets[self.socket_set].items():
				self.inputs.new(
					socket_def.socket_type.value,
					socket_def.label,
				)
				
				bl_socket = self.inputs[socket_def.label]
				socket_def.init(bl_socket)
		
		if self.socket_set in self.output_socket_sets:
			for socket_name, socket_def in self.output_socket_sets[self.socket_set].items():
				self.outputs.new(
					socket_def.socket_type.value,
					socket_def.label,
				)
				
				bl_socket = self.outputs[socket_def.label]
				socket_def.init(bl_socket)
		
		# Update "Previous"
		self.socket_set_previous = self.socket_set
	
	####################
	# - UI Methods
	####################
	def draw_buttons(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		"""This method draws the UI of the node itself.
		
		Specifically, it is used to expose the Presets dropdown.
		"""
		if self.preset is not None:
			layout.prop(self, "preset", text="")
		
		if hasattr(self, "socket_set"):
			layout.prop(self, "socket_set", text="")
		
		if hasattr(self, "draw_operators"):
			self.draw_operators(context, layout)
	
	####################
	# - Socket Getters
	####################
	def g_input_bl_socket(
		self,
		input_socket_name: contracts.SocketName,
	) -> bpy.types.NodeSocket:
		"""Returns the `bpy.types.NodeSocket` of an input socket by name.
		
		Args:
			input_socket_name: The name of the input socket, as defined in
				`self.input_sockets`.
		
		Returns:
			Blender's own node socket object.
		"""
		
		if input_socket_name in self.input_sockets:
			# Check Nicely that it Exists
			if input_socket_name not in self.input_sockets:
				msg = f"Input socket with name {input_socket_name} does not exist"
				raise ValueError(msg)
			
			return self.inputs[self.input_sockets[input_socket_name].label]
		
		elif hasattr(self, "input_socket_sets"):
			# You're on your own, chump
			
			return self.inputs[next(
				socket_def.label
				for socket_set, socket_dict in self.input_socket_sets.items()
				for socket_name, socket_def in socket_dict.items()
				if socket_name == input_socket_name
			)]
	
	def g_output_bl_socket(
		self,
		output_socket_name: contracts.SocketName,
	) -> bpy.types.NodeSocket:
		"""Returns the `bpy.types.NodeSocket` of an output socket by name.
		
		Args:
			output_socket_name: The name of the output socket, as defined in
				`self.output_sockets`.
		
		Returns:
			Blender's own node socket object.
		"""
		
		
		if output_socket_name in self.output_sockets:
			# (Guard) Socket Exists
			if output_socket_name not in self.output_sockets:
				msg = f"Input socket with name {output_socket_name} does not exist"
				raise ValueError(msg)
			
			return self.outputs[self.output_sockets[output_socket_name].label]
		
		elif hasattr(self, "input_socket_sets"):
			return self.outputs[next(
				socket_def.label
				for socket_set, socket_dict in self.input_socket_sets.items()
				for socket_name, socket_def in socket_dict.items()
				if socket_name == output_socket_name
			)]
	
	def g_output_socket_name(
		self,
		output_bl_socket_name: contracts.BLSocketName,
	) -> contracts.SocketName:
		if hasattr(self, "output_socket_sets"):
			return next(
				socket_name
				for socket_set, socket_dict in self.output_socket_sets.items()
				for socket_name, socket_def in socket_dict.items()
				if socket_def.label == output_bl_socket_name
			)
		else:
			return next(
				output_socket_name
				for output_socket_name in self.output_sockets.keys()
				if self.output_sockets[
					output_socket_name
				].label == output_bl_socket_name
			)
	
	####################
	# - Socket Setters
	####################
	def s_input_value(
		self,
		input_socket_name: contracts.SocketName,
		value: typ.Any,
	) -> None:
		"""Sets the value of an input socket, if the value is compatible with
		the socket.
		
		Args:
			input_socket_name: The name of the input socket.
			value: The value to set, which must be compatible with the
				socket.
		
		Raises:
			ValueError: If the value is incompatible with the socket, for
				example due to incompatible types, then a ValueError will be
				raised.
		"""
		bl_socket = self.g_input_bl_socket(input_socket_name)
		
		# Set the Value
		bl_socket.default_value = value
	
	####################
	# - Socket Computation
	####################
	def compute_input(
		self,
		input_socket_name: contracts.SocketName,
	) -> typ.Any:
		"""Computes the value of an input socket, by its name. Will
		automatically compute the output socket value of any linked
		nodes.
		
		Args:
			input_socket_name: The name of the input socket, as defined in
				`self.input_sockets`.
		"""
		bl_socket = self.g_input_bl_socket(input_socket_name)
		
		# Linked: Compute Output of Linked Socket
		if bl_socket.is_linked:
			linked_node = bl_socket.links[0].from_node
			
			# Compute the Linked Socket Name
			linked_bl_socket_name: contracts.BLSocketName = bl_socket.links[0].from_socket.name
			linked_socket_name = linked_node.g_output_socket_name(
				linked_bl_socket_name
			)
			
			# Compute the Linked Socket Value
			linked_socket_value = linked_node.compute_output(
				linked_socket_name
			)
			
			# (Guard) Check the Compatibility of the Linked Socket Value
			if not bl_socket.is_compatible(linked_socket_value):
				msg = f"Tried setting socket ({input_socket_name}) to incompatible value ({linked_socket_value}) of type {type(linked_socket_value)}"
				raise ValueError(msg)
			
			return linked_socket_value
		
		# Unlinked: Simply Retrieve Socket Value
		return bl_socket.default_value
		
	def compute_output(
		self,
		output_socket_name: contracts.SocketName,
	) -> typ.Any:
		"""Computes the value of an output socket name, from its socket name.
		
		Searches for methods decorated with `@computes_output_socket("name")`,
		which describe the computation that occurs to actually compute the
		value of an output socket from ex. input sockets and node properties.
		
		Args:
			output_socket_name: The name declaring the output socket,
				for which this method computes the output.
		
		Returns:
			The value of the output socket, as computed by the dedicated method
			registered using the `@computes_output_socket` decorator.
		"""
		# Lookup the Function that Computes the Output Socket
		## The decorator ALWAYS produces a FuncOutputSocket.
		## Thus, we merely need to find a FuncOutputSocket
		output_socket_func = next(
			method.output_socket_func
			for attr_name in dir(self)  ## Lookup self.*
			if isinstance(
				method := getattr(self, attr_name),
				FuncOutputSocket,
			)  
			if method.output_socket_name == output_socket_name
		)
		
		return output_socket_func(self)
