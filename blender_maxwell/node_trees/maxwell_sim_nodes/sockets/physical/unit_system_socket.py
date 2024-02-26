import typing as typ

import bpy
import sympy as sp
import pydantic as pyd

from .. import base
from ... import contracts

def contract_units_to_items(socket_type):
	return [
		(
			unit_key,
			str(unit),
			f"{socket_type}-compatible unit",
		)
		for unit_key, unit in contracts.SocketType_to_units[
			socket_type
		]["values"].items()
	]
def default_unit_key_for(socket_type):
	return contracts.SocketType_to_units[
		socket_type
	]["default"]

####################
# - Blender Socket
####################
class PhysicalUnitSystemBLSocket(base.BLSocket):
	socket_type = contracts.SocketType.PhysicalUnitSystem
	bl_label = "PhysicalUnitSystem"
	
	####################
	# - Properties
	####################
	show_definition: bpy.props.BoolProperty(
		name="Show Unit System Definition",
		description="Toggle to show unit system definition",
		default=False,
		update=(lambda self, context: self.trigger_updates()),
	)
		
	unit_time: bpy.props.EnumProperty(
		name="Time Unit",
		description="Unit of time",
		items=contract_units_to_items(contracts.SocketType.PhysicalTime),
		default=default_unit_key_for(contracts.SocketType.PhysicalTime),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	unit_angle: bpy.props.EnumProperty(
		name="Angle Unit",
		description="Unit of angle",
		items=contract_units_to_items(contracts.SocketType.PhysicalAngle),
		default=default_unit_key_for(contracts.SocketType.PhysicalAngle),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	unit_length: bpy.props.EnumProperty(
		name="Length Unit",
		description="Unit of length",
		items=contract_units_to_items(contracts.SocketType.PhysicalLength),
		default=default_unit_key_for(contracts.SocketType.PhysicalLength),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_area: bpy.props.EnumProperty(
		name="Area Unit",
		description="Unit of area",
		items=contract_units_to_items(contracts.SocketType.PhysicalArea),
		default=default_unit_key_for(contracts.SocketType.PhysicalArea),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_volume: bpy.props.EnumProperty(
		name="Volume Unit",
		description="Unit of time",
		items=contract_units_to_items(contracts.SocketType.PhysicalVolume),
		default=default_unit_key_for(contracts.SocketType.PhysicalVolume),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	unit_point_2d: bpy.props.EnumProperty(
		name="Point2D Unit",
		description="Unit of 2D points",
		items=contract_units_to_items(contracts.SocketType.PhysicalPoint2D),
		default=default_unit_key_for(contracts.SocketType.PhysicalPoint2D),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_point_3d: bpy.props.EnumProperty(
		name="Point3D Unit",
		description="Unit of 3D points",
		items=contract_units_to_items(contracts.SocketType.PhysicalPoint3D),
		default=default_unit_key_for(contracts.SocketType.PhysicalPoint3D),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	unit_size_2d: bpy.props.EnumProperty(
		name="Size2D Unit",
		description="Unit of 2D sizes",
		items=contract_units_to_items(contracts.SocketType.PhysicalSize2D),
		default=default_unit_key_for(contracts.SocketType.PhysicalSize2D),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_size_3d: bpy.props.EnumProperty(
		name="Size3D Unit",
		description="Unit of 3D sizes",
		items=contract_units_to_items(contracts.SocketType.PhysicalSize3D),
		default=default_unit_key_for(contracts.SocketType.PhysicalSize3D),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	unit_mass: bpy.props.EnumProperty(
		name="Mass Unit",
		description="Unit of mass",
		items=contract_units_to_items(contracts.SocketType.PhysicalMass),
		default=default_unit_key_for(contracts.SocketType.PhysicalMass),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	unit_speed: bpy.props.EnumProperty(
		name="Speed Unit",
		description="Unit of speed",
		items=contract_units_to_items(contracts.SocketType.PhysicalSpeed),
		default=default_unit_key_for(contracts.SocketType.PhysicalSpeed),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_accel_scalar: bpy.props.EnumProperty(
		name="Accel Unit",
		description="Unit of acceleration",
		items=contract_units_to_items(contracts.SocketType.PhysicalAccelScalar),
		default=default_unit_key_for(contracts.SocketType.PhysicalAccelScalar),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_force_scalar: bpy.props.EnumProperty(
		name="Force Scalar Unit",
		description="Unit of scalar force",
		items=contract_units_to_items(contracts.SocketType.PhysicalForceScalar),
		default=default_unit_key_for(contracts.SocketType.PhysicalForceScalar),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_accel_3d_vector: bpy.props.EnumProperty(
		name="Accel3D Unit",
		description="Unit of 3D vector acceleration",
		items=contract_units_to_items(contracts.SocketType.PhysicalAccel3DVector),
		default=default_unit_key_for(contracts.SocketType.PhysicalAccel3DVector),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_force_3d_vector: bpy.props.EnumProperty(
		name="Force3D Unit",
		description="Unit of 3D vector force",
		items=contract_units_to_items(contracts.SocketType.PhysicalForce3DVector),
		default=default_unit_key_for(contracts.SocketType.PhysicalForce3DVector),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	unit_freq: bpy.props.EnumProperty(
		name="Freq Unit",
		description="Unit of frequency",
		items=contract_units_to_items(contracts.SocketType.PhysicalFreq),
		default=default_unit_key_for(contracts.SocketType.PhysicalFreq),
		update=(lambda self, context: self.trigger_updates()),
	)
	unit_vac_wl: bpy.props.EnumProperty(
		name="VacWL Unit",
		description="Unit of vacuum wavelength",
		items=contract_units_to_items(contracts.SocketType.PhysicalVacWL),
		default=default_unit_key_for(contracts.SocketType.PhysicalVacWL),
		update=(lambda self, context: self.trigger_updates()),
	)
	
	####################
	# - UI
	####################
	def draw_label_row(self, label_col_row: bpy.types.UILayout, text) -> None:
		label_col_row.label(text=text)
		label_col_row.prop(self, "show_definition", toggle=True, text="", icon="MOD_LENGTH")
	
	def draw_value(self, col: bpy.types.UILayout) -> None:
		if self.show_definition:
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.prop(self, "unit_time", text="")
			col_row.prop(self, "unit_angle", text="")
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.prop(self, "unit_length", text="")
			col_row.prop(self, "unit_area", text="")
			col_row.prop(self, "unit_volume", text="")
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Point")
			col_row.prop(self, "unit_point_2d", text="")
			col_row.prop(self, "unit_point_3d", text="")
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Size")
			col_row.prop(self, "unit_size_2d", text="")
			col_row.prop(self, "unit_size_3d", text="")
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Mass")
			col_row.prop(self, "unit_mass", text="")
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Vel")
			col_row.prop(self, "unit_speed", text="")
			#col_row.prop(self, "unit_vel_2d_vector", text="")
			#col_row.prop(self, "unit_vel_3d_vector", text="")
			
			col_row=col.row(align=True)
			col_row.label(text="Accel")
			col_row.prop(self, "unit_accel_scalar", text="")
			#col_row.prop(self, "unit_accel_2d_vector", text="")
			col_row.prop(self, "unit_accel_3d_vector", text="")
			
			col_row=col.row(align=True)
			col_row.label(text="Force")
			col_row.prop(self, "unit_force_scalar", text="")
			#col_row.prop(self, "unit_force_2d_vector", text="")
			col_row.prop(self, "unit_force_3d_vector", text="")
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Freq")
			col_row.prop(self, "unit_freq", text="")
			
			col_row=col.row(align=True)
			col_row.alignment = "EXPAND"
			col_row.label(text="Vac WL")
			col_row.prop(self, "unit_vac_wl", text="")
	
	####################
	# - Default Value
	####################
	@property
	def default_value(self) -> sp.Expr:
		ST = contracts.SocketType
		SM = lambda socket_type: contracts.SocketType_to_units[
			socket_type
		]["values"]
		
		return {
			socket_type: SM(socket_type)[socket_unit_prop]
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
				(ST.PhysicalAccel3DVector, self.unit_accel_3d_vector),
				(ST.PhysicalForce3DVector, self.unit_force_3d_vector),
			
				(ST.PhysicalFreq, self.unit_freq),
				(ST.PhysicalVacWL, self.unit_vac_wl),
			]
		}
	
	@default_value.setter
	def default_value(self, value: typ.Any) -> None:
		pass

####################
# - Socket Configuration
####################
class PhysicalUnitSystemSocketDef(pyd.BaseModel):
	socket_type: contracts.SocketType = contracts.SocketType.PhysicalUnitSystem
	label: str
	
	show_by_default: bool = False
	
	def init(self, bl_socket: PhysicalUnitSystemBLSocket) -> None:
		bl_socket.show_definition = self.show_by_default

####################
# - Blender Registration
####################
BL_REGISTER = [
	PhysicalUnitSystemBLSocket,
]
