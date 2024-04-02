
import sympy as sp

from .. import contracts as ct
from .. import sockets
from . import base


class KitchenSinkNode(base.MaxwellSimNode):
	node_type = ct.NodeType.KitchenSink

	bl_label = 'Kitchen Sink'
	# bl_icon = ...

	####################
	# - Sockets
	####################
	input_sockets = {
		'Static Data': sockets.AnySocketDef(),
	}
	input_socket_sets = {
		'Basic': {
			'Any': sockets.AnySocketDef(),
			'Bool': sockets.BoolSocketDef(),
			'FilePath': sockets.FilePathSocketDef(),
			'Text': sockets.TextSocketDef(),
		},
		'Number': {
			'Integer': sockets.IntegerNumberSocketDef(),
			'Rational': sockets.RationalNumberSocketDef(),
			'Real': sockets.RealNumberSocketDef(),
			'Complex': sockets.ComplexNumberSocketDef(),
		},
		'Vector': {
			'Real 2D': sockets.Real2DVectorSocketDef(),
			'Real 3D': sockets.Real3DVectorSocketDef(
				default_value=sp.Matrix([0.0, 0.0, 0.0])
			),
			'Complex 2D': sockets.Complex2DVectorSocketDef(),
			'Complex 3D': sockets.Complex3DVectorSocketDef(),
		},
		'Physical': {
			'Time': sockets.PhysicalTimeSocketDef(),
			# "physical_point_2d": sockets.PhysicalPoint2DSocketDef(),
			'Angle': sockets.PhysicalAngleSocketDef(),
			'Length': sockets.PhysicalLengthSocketDef(),
			'Area': sockets.PhysicalAreaSocketDef(),
			'Volume': sockets.PhysicalVolumeSocketDef(),
			'Point 3D': sockets.PhysicalPoint3DSocketDef(),
			##"physical_size_2d": sockets.PhysicalSize2DSocketDef(),
			'Size 3D': sockets.PhysicalSize3DSocketDef(),
			'Mass': sockets.PhysicalMassSocketDef(),
			'Speed': sockets.PhysicalSpeedSocketDef(),
			'Accel Scalar': sockets.PhysicalAccelScalarSocketDef(),
			'Force Scalar': sockets.PhysicalForceScalarSocketDef(),
			# "physical_accel_3dvector": sockets.PhysicalAccel3DVectorSocketDef(),
			##"physical_force_3dvector": sockets.PhysicalForce3DVectorSocketDef(),
			'Pol': sockets.PhysicalPolSocketDef(),
			'Freq': sockets.PhysicalFreqSocketDef(),
		},
		'Blender': {
			'Object': sockets.BlenderObjectSocketDef(),
			'Collection': sockets.BlenderCollectionSocketDef(),
			'Image': sockets.BlenderImageSocketDef(),
			'GeoNodes': sockets.BlenderGeoNodesSocketDef(),
			'Text': sockets.BlenderTextSocketDef(),
		},
		'Maxwell': {
			'Source': sockets.MaxwellSourceSocketDef(),
			'Temporal Shape': sockets.MaxwellTemporalShapeSocketDef(),
			'Medium': sockets.MaxwellMediumSocketDef(),
			'Medium Non-Linearity': sockets.MaxwellMediumNonLinearitySocketDef(),
			'Structure': sockets.MaxwellStructureSocketDef(),
			'Bound Box': sockets.MaxwellBoundBoxSocketDef(),
			'Bound Face': sockets.MaxwellBoundFaceSocketDef(),
			'Monitor': sockets.MaxwellMonitorSocketDef(),
			'FDTD Sim': sockets.MaxwellFDTDSimSocketDef(),
			'Sim Grid': sockets.MaxwellSimGridSocketDef(),
			'Sim Grid Axis': sockets.MaxwellSimGridAxisSocketDef(),
		},
	}

	output_sockets = {
		'Static Data': sockets.AnySocketDef(),
	}
	output_socket_sets = input_socket_sets


####################
# - Blender Registration
####################
BL_REGISTER = [
	KitchenSinkNode,
]
BL_NODES = {ct.NodeType.KitchenSink: (ct.NodeCategory.MAXWELLSIM_INPUTS)}
