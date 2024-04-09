#import typing as typ
#import bpy
#
#from .. import contracts as ct
#
#
#
#class MaxwellSimProp(bpy.types.PropertyGroup):
#	"""A Blender property usable in nodes and sockets."""
#	name: str = ""
#	data_flow_kind: ct.DataFlowKind
#
#	value: dict[str, tuple[bpy.types.Property, dict]] | None = None
#
#	def __init_subclass__(cls, **kwargs: typ.Any):
#		log.debug('Initializing Prop: %s', cls.node_type)
#		super().__init_subclass__(**kwargs)
#
#		# Setup Value
#		if cls.value:
#			cls.__annotations__['raw_value'] = value
#
#
#	@property
#	def value(self):
#		if self.data_flow_kind
