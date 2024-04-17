import bpy
import sympy as sp
import sympy.physics.units as spu

from .....utils import extra_sympy_units as spux
from .....utils import logger
from .....utils.pydantic_sympy import SympyExpr
from ... import contracts as ct
from .. import base

log = logger.get(__name__)


####################
# - Blender Socket
####################
class PhysicalLengthBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.PhysicalLength
	bl_label = 'PhysicalLength'
	use_units = True

	####################
	# - Properties
	####################
	raw_value: bpy.props.FloatProperty(
		name='Unitless Length',
		description='Represents the unitless part of the length',
		default=0.0,
		precision=6,
		update=(lambda self, context: self.sync_prop('raw_value', context)),
	)

	min_len: bpy.props.FloatProperty(
		name='Min Length',
		description='Lowest length',
		default=0.0,
		precision=4,
		update=(lambda self, context: self.sync_prop('min_len', context)),
	)
	max_len: bpy.props.FloatProperty(
		name='Max Length',
		description='Highest length',
		default=0.0,
		precision=4,
		update=(lambda self, context: self.sync_prop('max_len', context)),
	)
	steps: bpy.props.IntProperty(
		name='Length Steps',
		description='# of steps between min and max',
		default=2,
		update=(lambda self, context: self.sync_prop('steps', context)),
	)

	####################
	# - Socket UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'raw_value', text='')

	def draw_lazy_value_range(self, col: bpy.types.UILayout) -> None:
		col.prop(self, 'min_len', text='Min')
		col.prop(self, 'max_len', text='Max')
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
	def lazy_value_range(self) -> ct.LazyDataValueRange:
		return ct.LazyDataValueRange(
			symbols=set(),
			has_unit=True,
			unit=self.unit,
			start=sp.S(self.min_len) * self.unit,
			stop=sp.S(self.max_len) * self.unit,
			steps=self.steps,
			scaling='lin',
		)

	@lazy_value_range.setter
	def lazy_value_range(self, value: tuple[sp.Expr, sp.Expr, int]) -> None:
		self.min_len = spux.sympy_to_python(spux.scale_to_unit(value[0], self.unit))
		self.max_len = spux.sympy_to_python(spux.scale_to_unit(value[1], self.unit))
		self.steps = value[2]


####################
# - Socket Configuration
####################
class PhysicalLengthSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.PhysicalLength
	is_array: bool = False

	default_value: SympyExpr = 1 * spu.um
	default_unit: SympyExpr | None = None

	min_len: SympyExpr = 400.0 * spu.nm
	max_len: SympyExpr = 700.0 * spu.nm
	steps: SympyExpr = 50

	def init(self, bl_socket: PhysicalLengthBLSocket) -> None:
		if self.default_unit:
			bl_socket.unit = self.default_unit

		bl_socket.value = self.default_value
		if self.is_array:
			bl_socket.active_kind = ct.FlowKind.LazyValueRange
			bl_socket.lazy_value_range = (self.min_len, self.max_len, self.steps)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalLengthBLSocket,
]
