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
			"physical_angle": sockets.PhysicalAngleSocketDef(label="PhysicalAngle"),
			"physical_length": sockets.PhysicalLengthSocketDef(label="PhysicalLength"),
			"physical_area": sockets.PhysicalAreaSocketDef(label="PhysicalArea"),
			"physical_volume": sockets.PhysicalVolumeSocketDef(label="PhysicalVolume"),
			"physical_mass": sockets.PhysicalMassSocketDef(label="PhysicalMass"),
			"physical_speed": sockets.PhysicalSpeedSocketDef(label="PhysicalSpeed"),
			"physical_accel": sockets.PhysicalAccelSocketDef(label="PhysicalAccel"),
			"physical_force": sockets.PhysicalForceSocketDef(label="PhysicalForce"),
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
			"maxwell_structure": sockets.MaxwellMediumSocketDef(label="MaxwellMedium"),
			"maxwell_bound_box": sockets.MaxwellBoundBoxSocketDef(label="MaxwellBoundBox"),
			"maxwell_bound_face": sockets.MaxwellBoundFaceSocketDef(label="MaxwellBoundFace"),
			"maxwell_monitor": sockets.MaxwellMonitorSocketDef(label="MaxwellMonitor"),
			"maxwell_monitor": sockets.MaxwellFDTDSimSocketDef(label="MaxwellFDTDSim"),
			"maxwell_monitor": sockets.MaxwellSimGridSocketDef(label="MaxwellSimGrid"),
			"maxwell_monitor": sockets.MaxwellSimGridAxisSocketDef(label="MaxwellSimGridAxis"),
		},
	}
	
	output_sockets = {}
	output_socket_sets = input_socket_sets



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
