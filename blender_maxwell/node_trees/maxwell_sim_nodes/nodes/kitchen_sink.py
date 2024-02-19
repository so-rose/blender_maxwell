import tidy3d as td
import sympy as sp
import sympy.physics.units as spu

from .. import contracts
from .. import sockets
from . import base

class KitchenSinkNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.KitchenSink
	
	bl_label = "Kitchen Sink"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {}
	input_socket_sets = {
		"basic": {
			"basic_any": sockets.AnySocketDef(label="Any"),
			"basic_bool": sockets.BoolSocketDef(label="Bool"),
			"basic_filepath": sockets.FilePathSocketDef(label="FilePath"),
			"basic_text": sockets.TextSocketDef(label="Text"),
		},
		"number": {
			"number_integer": sockets.IntegerNumberSocketDef(label="IntegerNumber"),
			"number_rational": sockets.RationalNumberSocketDef(label="RationalNumber"),
			"number_real": sockets.RealNumberSocketDef(label="RealNumber"),
			"number_complex": sockets.ComplexNumberSocketDef(label="ComplexNumber"),
		},
		"vector": {
			"vector_real2dvector": sockets.Real2DVectorSocketDef(label="Real2DVector"),
			"vector_complex2dvector": sockets.Complex2DVectorSocketDef(label="Complex2DVector"),
			"vector_real3dvector": sockets.Real3DVectorSocketDef(label="Real3DVector"),
			"vector_complex3dvector": sockets.Complex3DVectorSocketDef(label="Complex3DVector"),
		},
		"physical": {
			"physical_time": sockets.PhysicalTimeSocketDef(label="PhysicalTime"),
			#"physical_point_2d": sockets.PhysicalPoint2DSocketDef(label="PhysicalPoint2D"),
			"physical_angle": sockets.PhysicalAngleSocketDef(label="PhysicalAngle"),
			"physical_length": sockets.PhysicalLengthSocketDef(label="PhysicalLength"),
			"physical_area": sockets.PhysicalAreaSocketDef(label="PhysicalArea"),
			"physical_volume": sockets.PhysicalVolumeSocketDef(label="PhysicalVolume"),
			"physical_point_3d": sockets.PhysicalPoint3DSocketDef(label="PhysicalPoint3D"),
			#"physical_size_2d": sockets.PhysicalSize2DSocketDef(label="PhysicalSize2D"),
			"physical_size_3d": sockets.PhysicalSize3DSocketDef(label="PhysicalSize3D"),
			"physical_mass": sockets.PhysicalMassSocketDef(label="PhysicalMass"),
			"physical_speed": sockets.PhysicalSpeedSocketDef(label="PhysicalSpeed"),
			"physical_accel_scalar": sockets.PhysicalAccelScalarSocketDef(label="PhysicalAccelScalar"),
			"physical_force_scalar": sockets.PhysicalForceScalarSocketDef(label="PhysicalForceScalar"),
			#"physical_accel_3dvector": sockets.PhysicalAccel3DVectorSocketDef(label="PhysicalAccel3DVector"),
			#"physical_force_3dvector": sockets.PhysicalForce3DVectorSocketDef(label="PhysicalForce3DVector"),
			"physical_pol": sockets.PhysicalPolSocketDef(label="PhysicalPol"),
			"physical_freq": sockets.PhysicalFreqSocketDef(label="PhysicalFreq"),
			"physical_spec_power_dist": sockets.PhysicalSpecPowerDistSocketDef(label="PhysicalSpecPowerDist"),
			"physical_spec_rel_perm_dist": sockets.PhysicalSpecRelPermDistSocketDef(label="PhysicalSpecRelPermDist"),
		},
		"blender": {
			"blender_object": sockets.BlenderObjectSocketDef(label="BlenderObject"),
			"blender_collection": sockets.BlenderCollectionSocketDef(label="BlenderCollection"),
			"blender_image": sockets.BlenderImageSocketDef(label="BlenderImage"),
			"blender_volume": sockets.BlenderVolumeSocketDef(label="BlenderVolume"),
			"blender_geonodes": sockets.BlenderGeoNodesSocketDef(label="BlenderGeoNodes"),
			"blender_text": sockets.BlenderTextSocketDef(label="BlenderText"),
		},
		"maxwell": {
			"maxwell_source": sockets.MaxwellSourceSocketDef(label="MaxwellSource"),
			"maxwell_temporal_shape": sockets.MaxwellTemporalShapeSocketDef(label="MaxwellTemporalShape"),
			"maxwell_medium": sockets.MaxwellMediumSocketDef(label="MaxwellMedium"),
			#"maxwell_medium_nonlinearity": sockets.MaxwellMediumNonLinearitySocketDef(label="MaxwellMediumNonLinearity"),
			"maxwell_structure": sockets.MaxwellStructureSocketDef(label="MaxwellMedium"),
			"maxwell_bound_box": sockets.MaxwellBoundBoxSocketDef(label="MaxwellBoundBox"),
			"maxwell_bound_face": sockets.MaxwellBoundFaceSocketDef(label="MaxwellBoundFace"),
			"maxwell_monitor": sockets.MaxwellMonitorSocketDef(label="MaxwellMonitor"),
			"maxwell_fdtd_sim": sockets.MaxwellFDTDSimSocketDef(label="MaxwellFDTDSim"),
			"maxwell_sim_grid": sockets.MaxwellSimGridSocketDef(label="MaxwellSimGrid"),
			"maxwell_sim_grid_axis": sockets.MaxwellSimGridAxisSocketDef(label="MaxwellSimGridAxis"),
		},
	}
	
	output_sockets = {}
	output_socket_sets = {
		k + " Output": v
		for k, v in input_socket_sets.items()
	}



####################
# - Blender Registration
####################
BL_REGISTER = [
	KitchenSinkNode,
]
BL_NODES = {
	contracts.NodeType.KitchenSink: (
		contracts.NodeCategory.MAXWELLSIM_INPUTS
	)
}
