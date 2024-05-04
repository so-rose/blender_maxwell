import bpy

# from blender_maxwell.utils.pydantic_sympy import SympyExpr
from ... import contracts as ct
from .. import base

ST = ct.SocketType


def SU(socket_type):  # noqa: N802, D103
	return ct.SOCKET_UNITS[socket_type]['values']


####################
# - Utilities
####################
def contract_units_to_items(socket_type: ST) -> list[tuple[str, str, str]]:
	"""For a given socket type, get a bpy.props.EnumProperty-compatible list-tuple of items to display in the enum dropdown menu."""
	return [
		(
			unit_key,
			str(unit),
			f'{socket_type}-compatible unit',
		)
		for unit_key, unit in SU(socket_type).items()
	]


def default_unit_key_for(socket_type: ST) -> str:
	"""For a given socket type, return the default key corresponding to the default unit."""
	return ct.SOCKET_UNITS[socket_type]['default']


####################
# - Blender Socket
####################
class PhysicalUnitSystemBLSocket(base.MaxwellSimSocket):
	socket_type = ST.PhysicalUnitSystem
	bl_label = 'Unit System'

	####################
	# - Properties
	####################
	show_definition: bpy.props.BoolProperty(
		name='Show Unit System Definition',
		description='Toggle to show unit system definition',
		default=False,
		update=(lambda self, context: self.on_prop_changed('show_definition', context)),
	)

	unit_time: bpy.props.EnumProperty(
		name='Time Unit',
		description='Unit of time',
		items=contract_units_to_items(ST.PhysicalTime),
		default=default_unit_key_for(ST.PhysicalTime),
		update=(lambda self, context: self.on_prop_changed('unit_time', context)),
	)

	unit_angle: bpy.props.EnumProperty(
		name='Angle Unit',
		description='Unit of angle',
		items=contract_units_to_items(ST.PhysicalAngle),
		default=default_unit_key_for(ST.PhysicalAngle),
		update=(lambda self, context: self.on_prop_changed('unit_angle', context)),
	)

	unit_length: bpy.props.EnumProperty(
		name='Length Unit',
		description='Unit of length',
		items=contract_units_to_items(ST.PhysicalLength),
		default=default_unit_key_for(ST.PhysicalLength),
		update=(lambda self, context: self.on_prop_changed('unit_length', context)),
	)
	unit_area: bpy.props.EnumProperty(
		name='Area Unit',
		description='Unit of area',
		items=contract_units_to_items(ST.PhysicalArea),
		default=default_unit_key_for(ST.PhysicalArea),
		update=(lambda self, context: self.on_prop_changed('unit_area', context)),
	)
	unit_volume: bpy.props.EnumProperty(
		name='Volume Unit',
		description='Unit of time',
		items=contract_units_to_items(ST.PhysicalVolume),
		default=default_unit_key_for(ST.PhysicalVolume),
		update=(lambda self, context: self.on_prop_changed('unit_volume', context)),
	)

	unit_point_2d: bpy.props.EnumProperty(
		name='Point2D Unit',
		description='Unit of 2D points',
		items=contract_units_to_items(ST.PhysicalPoint2D),
		default=default_unit_key_for(ST.PhysicalPoint2D),
		update=(lambda self, context: self.on_prop_changed('unit_point_2d', context)),
	)
	unit_point_3d: bpy.props.EnumProperty(
		name='Point3D Unit',
		description='Unit of 3D points',
		items=contract_units_to_items(ST.PhysicalPoint3D),
		default=default_unit_key_for(ST.PhysicalPoint3D),
		update=(lambda self, context: self.on_prop_changed('unit_point_3d', context)),
	)

	unit_size_2d: bpy.props.EnumProperty(
		name='Size2D Unit',
		description='Unit of 2D sizes',
		items=contract_units_to_items(ST.PhysicalSize2D),
		default=default_unit_key_for(ST.PhysicalSize2D),
		update=(lambda self, context: self.on_prop_changed('unit_size_2d', context)),
	)
	unit_size_3d: bpy.props.EnumProperty(
		name='Size3D Unit',
		description='Unit of 3D sizes',
		items=contract_units_to_items(ST.PhysicalSize3D),
		default=default_unit_key_for(ST.PhysicalSize3D),
		update=(lambda self, context: self.on_prop_changed('unit_size_3d', context)),
	)

	unit_mass: bpy.props.EnumProperty(
		name='Mass Unit',
		description='Unit of mass',
		items=contract_units_to_items(ST.PhysicalMass),
		default=default_unit_key_for(ST.PhysicalMass),
		update=(lambda self, context: self.on_prop_changed('unit_mass', context)),
	)

	unit_speed: bpy.props.EnumProperty(
		name='Speed Unit',
		description='Unit of speed',
		items=contract_units_to_items(ST.PhysicalSpeed),
		default=default_unit_key_for(ST.PhysicalSpeed),
		update=(lambda self, context: self.on_prop_changed('unit_speed', context)),
	)
	unit_accel_scalar: bpy.props.EnumProperty(
		name='Accel Unit',
		description='Unit of acceleration',
		items=contract_units_to_items(ST.PhysicalAccelScalar),
		default=default_unit_key_for(ST.PhysicalAccelScalar),
		update=(
			lambda self, context: self.on_prop_changed('unit_accel_scalar', context)
		),
	)
	unit_force_scalar: bpy.props.EnumProperty(
		name='Force Scalar Unit',
		description='Unit of scalar force',
		items=contract_units_to_items(ST.PhysicalForceScalar),
		default=default_unit_key_for(ST.PhysicalForceScalar),
		update=(
			lambda self, context: self.on_prop_changed('unit_force_scalar', context)
		),
	)
	unit_accel_3d: bpy.props.EnumProperty(
		name='Accel3D Unit',
		description='Unit of 3D vector acceleration',
		items=contract_units_to_items(ST.PhysicalAccel3D),
		default=default_unit_key_for(ST.PhysicalAccel3D),
		update=(lambda self, context: self.on_prop_changed('unit_accel_3d', context)),
	)
	unit_force_3d: bpy.props.EnumProperty(
		name='Force3D Unit',
		description='Unit of 3D vector force',
		items=contract_units_to_items(ST.PhysicalForce3D),
		default=default_unit_key_for(ST.PhysicalForce3D),
		update=(lambda self, context: self.on_prop_changed('unit_force_3d', context)),
	)

	unit_freq: bpy.props.EnumProperty(
		name='Freq Unit',
		description='Unit of frequency',
		items=contract_units_to_items(ST.PhysicalFreq),
		default=default_unit_key_for(ST.PhysicalFreq),
		update=(lambda self, context: self.on_prop_changed('unit_freq', context)),
	)

	####################
	# - UI
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text) -> None:
		row.label(text=text)
		row.prop(self, 'show_definition', toggle=True, text='', icon='MOD_LENGTH')

	def draw_value(self, col: bpy.types.UILayout) -> None:
		if self.show_definition:
			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Time | Angle')
			col.prop(self, 'unit_time', text='t')
			col.prop(self, 'unit_angle', text='θ')
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Len | Area | Vol')
			col.prop(self, 'unit_length', text='l')
			col.prop(self, 'unit_area', text='l²')
			col.prop(self, 'unit_volume', text='l³')
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Point')
			col.prop(self, 'unit_point_2d', text='P₂')
			col.prop(self, 'unit_point_3d', text='P₃')
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Size')
			col.prop(self, 'unit_size_2d', text='S₂')
			col.prop(self, 'unit_size_3d', text='S₃')
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Mass')
			col.prop(self, 'unit_mass', text='M')
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Vel')
			col.prop(self, 'unit_speed', text='|v|')
			# col.prop(self, "unit_vel_2d_vector", text="")
			# col.prop(self, "unit_vel_3d_vector", text="")
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Accel')
			col.prop(self, 'unit_accel_scalar', text='|a|')
			# col.prop(self, "unit_accel_2d_vector", text="")
			col.prop(self, 'unit_accel_3d', text='a₃')
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Force')
			col.prop(self, 'unit_force_scalar', text='|F|')
			# col.prop(self, "unit_force_2d_vector", text="")
			col.prop(self, 'unit_force_3d', text='F₃')
			col.separator(factor=1.0)

			row = col.row()
			row.alignment = 'CENTER'
			row.label(text='Freq')
			col.prop(self, 'unit_freq', text='t⁽⁻¹⁾')

	####################
	# - Default Value
	####################
	@property
	def value(self) -> dict[ST, SympyExpr]:
		return {
			socket_type: SU(socket_type)[socket_unit_prop]
			for socket_type, socket_unit_prop in [
				(ST.PhysicalTime, self.unit_time),
				(ST.PhysicalAngle, self.unit_angle),
				(ST.PhysicalLength, self.unit_length),
				(ST.PhysicalArea, self.unit_area),
				(ST.PhysicalVolume, self.unit_volume),
				(ST.PhysicalPoint2D, self.unit_point_2d),
				(ST.PhysicalPoint3D, self.unit_point_3d),
				(ST.PhysicalSize2D, self.unit_size_2d),
				(ST.PhysicalSize3D, self.unit_size_3d),
				(ST.PhysicalMass, self.unit_mass),
				(ST.PhysicalSpeed, self.unit_speed),
				(ST.PhysicalAccelScalar, self.unit_accel_scalar),
				(ST.PhysicalForceScalar, self.unit_force_scalar),
				(ST.PhysicalAccel3D, self.unit_accel_3d),
				(ST.PhysicalForce3D, self.unit_force_3d),
				(ST.PhysicalFreq, self.unit_freq),
			]
		}


####################
# - Socket Configuration
####################
class PhysicalUnitSystemSocketDef(base.SocketDef):
	socket_type: ST = ST.PhysicalUnitSystem

	show_by_default: bool = False

	def init(self, bl_socket: PhysicalUnitSystemBLSocket) -> None:
		bl_socket.show_definition = self.show_by_default


####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalUnitSystemBLSocket,
]
