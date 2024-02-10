import numpy as np
import bpy
import nodeitems_utils

from .. import types

####################
# - Decorator: Output Socket
####################
def output_socket_cb(name):
	def decorator(func):
		func._output_socket_name = name  # Set a marker attribute
		return func
	return decorator

####################
# - Socket Type Casters
####################
SOCKET_CAST_MAP = {
	"NodeSocketBool": bool,
	"NodeSocketFloat": float,
	"NodeSocketFloatAngle": float,
	"NodeSocketFloatDistance": float,
	"NodeSocketFloatFactor": float,
	"NodeSocketFloatPercentage": float,
	"NodeSocketFloatTime": float,
	"NodeSocketFloatTimeAbsolute": float,
	"NodeSocketFloatUnsigned": float,
	"NodeSocketFloatInt": int,
	"NodeSocketFloatIntFactor": int,
	"NodeSocketFloatIntPercentage": int,
	"NodeSocketFloatIntUnsigned": int,
	"NodeSocketString": str,
	"NodeSocketVector": np.array,
	"NodeSocketVectorAcceleration": np.array,
	"NodeSocketVectorDirection": np.array,
	"NodeSocketVectorTranslation": np.array,
	"NodeSocketVectorVelocity": np.array,
	"NodeSocketVectorXYZ": np.array,
}

####################
# - Node Superclass
####################
def set_preset(self, context):
	for preset_name, preset_dict in self.socket_presets["_values"].items():
		if self.preset == preset_name:
			for input_socket_name, value in preset_dict.items():
				self.inputs[
					self.input_sockets[input_socket_name][1]
				].default_value = value
			
			break

class MaxwellSimTreeNode(bpy.types.Node):
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		required_attrs = [
			'bl_idname',
			'bl_label',
			'bl_icon',
			'input_sockets',
			'output_sockets',
		]
		for attr in required_attrs:
			if getattr(cls, attr, None) is None:
				raise TypeError(
					f"class {cls.__name__} is missing required '{attr}' attribute"
				)
		
		if hasattr(cls, 'socket_presets'):
			cls.__annotations__["preset"] = bpy.props.EnumProperty(
				name="Presets",
				description="Select a preset",
				items=cls.socket_presets["_description"],
				default=cls.socket_presets["_default"],
				update=set_preset,
			)
	
	####################
	# - Node Initialization
	####################
	def init(self, context):
		for input_socket_name in self.input_sockets:
			self.inputs.new(*self.input_sockets[input_socket_name][:2])
		
		for output_socket_name in self.output_sockets:
			self.outputs.new(*self.output_sockets[output_socket_name][:2])
		
		set_preset(self, context)
	
	####################
	# - Node Computation
	####################
	def compute_input(self, input_socket_name: str):
		"""Computes the value of an input socket name.
		"""
		bl_socket_type = self.input_sockets[input_socket_name][0]
		bl_socket = self.inputs[self.input_sockets[input_socket_name][1]]
		
		if bl_socket.is_linked:
			linked_node = bl_socket.links[0].from_node
			linked_bl_socket_name = bl_socket.links[0].from_socket.name
			result = linked_node.compute_output(linked_bl_socket_name)
		else:
			result = bl_socket.default_value
		
		if bl_socket_type in SOCKET_CAST_MAP:
			return SOCKET_CAST_MAP[bl_socket_type](result)
		
		return result
		
	def compute_output(self, output_bl_socket_name: str):
		"""Computes the value of an output socket name, from its Blender display name.
		"""
		output_socket_name = next(
			output_socket_name
			for output_socket_name in self.output_sockets.keys()
			if self.output_sockets[output_socket_name][1] == output_bl_socket_name
		)
		
		output_socket_name_to_cb = {
			getattr(attr, '_output_socket_name'): attr
			for attr_name in dir(self)
			if (
				callable(attr := getattr(self, attr_name))
				and hasattr(attr, '_output_socket_name')
			)
		}
		
		return output_socket_name_to_cb[output_socket_name]()
	
	####################
	# - Blender Configuration
	####################
	@classmethod
	def poll(cls, ntree):
		"""Constrain node instantiation to within a MaxwellSimTree."""
		
		return ntree.bl_idname == types.TreeType.MaxwellSim
