import bpy
import sympy as sp

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger
from blender_maxwell.utils.pydantic_sympy import SympyExpr

from ... import contracts as ct
from .. import base

log = logger.get(__name__)


####################
# - Blender Socket
####################
class PhysicalFreqBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalFreq
	bl_label = 'Frequency'
	use_units = True

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name='Unitless Frequency',
		description='Represents the unitless part of the frequency',
		default=0.0,
		precision=6,
		update=(lambda self, context: self.on_prop_changed('raw_value', context)),
	)

	min_freq: bpy.props.FloatProperty(
		name='Min Frequency',
		description='Lowest frequency',
		default=0.0,
		precision=4,
		update=(lambda self, context: self.on_prop_changed('min_freq', context)),
	)
	max_freq: bpy.props.FloatProperty(
		name='Max Frequency',
		description='Highest frequency',
		default=0.0,
		precision=4,
		update=(lambda self, context: self.on_prop_changed('max_freq', context)),
	)
	steps: bpy.props.IntProperty(
		name='Frequency Steps',
		description='# of steps between min and max',
		default=2,
		update=(lambda self, context: self.on_prop_changed('steps', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	def draw_lazy_value_range(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'min_freq', text='Min')
		col.prop(self, 'max_freq', text='Max')
		col.prop(self, 'steps', text='Steps')

	####################
	# - Default Value
	####################
	@property
	def value(self) -> SympyExpr:
		return self.raw_value * self.unit

	@value.setter
	def value(self, value: SympyExpr) -> None:
		self.raw_value = spux.sympy_to_python(spux.scale_to_unit(value, self.unit))

	@property
	def lazy_array_range(self) -> ct.LazyArrayRangeFlow:
		return ct.LazyArrayRangeFlow(
			symbols=set(),
			unit=self.unit,
			start=sp.S(self.min_freq) * self.unit,
			stop=sp.S(self.max_freq) * self.unit,
			steps=self.steps,
			scaling='lin',
		)

	@lazy_array_range.setter
	def lazy_array_range(self, value: ct.LazyArrayRangeFlow) -> None:
		self.min_freq = spux.sympy_to_python(
			spux.scale_to_unit(value.start * value.unit, self.unit)
		)
		self.max_freq = spux.sympy_to_python(
			spux.scale_to_unit(value.stop * value.unit, self.unit)
		)
		self.steps = value.steps


####################
# - Socket Configuration
####################
class PhysicalFreqSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.PhysicalFreq
	is_array: bool = False

	default_value: SympyExpr = 500 * spux.terahertz
	default_unit: SympyExpr = spux.terahertz

	min_freq: SympyExpr = 400.0 * spux.terahertz
	max_freq: SympyExpr = 600.0 * spux.terahertz
	steps: int = 50

	def init(self, bl_socket: PhysicalFreqBLSocket) -> None:
		bl_socket.unit = self.default_unit

		bl_socket.value = self.default_value
		if self.is_array:
			bl_socket.active_kind = ct.FlowKind.LazyArrayRange
			bl_socket.lazy_value_range = (self.min_freq, self.max_freq, self.steps)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalFreqBLSocket,
]
