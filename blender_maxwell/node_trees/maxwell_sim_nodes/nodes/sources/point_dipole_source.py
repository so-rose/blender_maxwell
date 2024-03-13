import typing as typ
import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

import bpy

from ... import contracts as ct
from ... import sockets
from .. import base
from ... import managed_objs

class PointDipoleSourceNode(base.MaxwellSimNode):
	node_type = ct.NodeType.PointDipoleSource
	bl_label = "Point Dipole Source"
	
	####################
	# - Sockets
	####################
	input_sockets = {
		"Temporal Shape": sockets.MaxwellTemporalShapeSocketDef(),
		"Center": sockets.PhysicalPoint3DSocketDef(),
		"Interpolate": sockets.BoolSocketDef(
			default_value=True,
		),
	}
	output_sockets = {
		"Source": sockets.MaxwellSourceSocketDef(),
	}
	
	managed_obj_defs = {
		"sphere_empty": ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLObject(name),
			name_prefix="",
		)
	}
	
	####################
	# - Properties
	####################
	pol_axis: bpy.props.EnumProperty(
		name="Polarization Axis",
		description="Polarization Axis",
		items=[
			("EX", "Ex", "Electric field in x-dir"),
			("EY", "Ey", "Electric field in y-dir"),
			("EZ", "Ez", "Electric field in z-dir"),
		],
		default="EX",
		update=(lambda self, context: self.sync_prop("pol_axis", context)),
	)
	
	####################
	# - UI
	####################
	def draw_props(self, context, layout):
		split = layout.split(factor=0.6)
		
		col = split.column()
		col.label(text="Pol Axis")
		
		col = split.column()
		col.prop(self, "pol_axis", text="")
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket(
		"Source",
		input_sockets={"Temporal Shape", "Center", "Interpolate"},
		props={"pol_axis"},
	)
	def compute_source(self, input_sockets: dict[str, typ.Any], props: dict[str, typ.Any]) -> td.PointDipole:
		pol_axis = {
			"EX": "Ex",
			"EY": "Ey",
			"EZ": "Ez",
		}[props["pol_axis"]]
		
		temporal_shape = input_sockets["Temporal Shape"]
		_center = input_sockets["Center"]
		interpolate = input_sockets["Interpolate"]
		
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		
		_res = td.PointDipole(
			center=center,
			source_time=temporal_shape,
			interpolate=interpolate,
			polarization=pol_axis,
		)
		return _res
	
	####################
	# - Preview
	####################
	@base.on_value_changed(
		socket_name="Center",
		input_sockets={"Center"},
		managed_objs={"sphere_empty"},
	)
	def on_value_changed__center(
		self,
		input_sockets: dict,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		_center = input_sockets["Center"]
		center = tuple(spu.convert_to(_center, spu.um) / spu.um)
		## TODO: Preview unit system?? Presume um for now
		
		mobj = managed_objs["sphere_empty"]
		bl_object = mobj.bl_object("EMPTY")
		bl_object.location = center #tuple([float(el) for el in center])
	
	@base.on_show_preview(
		managed_objs={"sphere_empty"},
	)
	def on_show_preview(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
	):
		managed_objs["sphere_empty"].show_preview(
			"EMPTY",
			empty_display_type="SPHERE",
		)
		managed_objs["sphere_empty"].bl_object("EMPTY").empty_display_size = 0.2



####################
# - Blender Registration
####################
BL_REGISTER = [
	PointDipoleSourceNode,
]
BL_NODES = {
	ct.NodeType.PointDipoleSource: (
		ct.NodeCategory.MAXWELLSIM_SOURCES
	)
}
