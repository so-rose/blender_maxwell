from .socket_types import SocketType as ST

SOCKET_SHAPES = {
	# Basic
	ST.Any: "CIRCLE",
	ST.Bool: "CIRCLE",
	ST.Text: "SQUARE",
	ST.FilePath: "SQUARE",
	ST.Secret: "SQUARE",
	
	# Number
	ST.IntegerNumber: "CIRCLE",
	ST.RationalNumber: "CIRCLE",
	ST.RealNumber: "CIRCLE",
	ST.ComplexNumber: "CIRCLE_DOT",
	
	# Vector
	ST.Real2DVector: "SQUARE_DOT",
	ST.Complex2DVector: "DIAMOND_DOT",
	ST.Real3DVector: "SQUARE_DOT",
	ST.Complex3DVector: "DIAMOND_DOT",
	
	# Physical
	ST.PhysicalUnitSystem: "CIRCLE",
	ST.PhysicalTime: "CIRCLE",
	ST.PhysicalAngle: "DIAMOND",
	ST.PhysicalLength: "SQUARE",
	ST.PhysicalArea: "SQUARE",
	ST.PhysicalVolume: "SQUARE",
	ST.PhysicalPoint2D: "DIAMOND",
	ST.PhysicalPoint3D: "DIAMOND",
	ST.PhysicalSize2D: "SQUARE",
	ST.PhysicalSize3D: "SQUARE",
	ST.PhysicalMass: "CIRCLE",
	ST.PhysicalSpeed: "CIRCLE",
	ST.PhysicalAccelScalar: "CIRCLE",
	ST.PhysicalForceScalar: "CIRCLE",
	ST.PhysicalAccel3DVector: "SQUARE_DOT",
	ST.PhysicalForce3DVector: "SQUARE_DOT",
	ST.PhysicalPol: "DIAMOND",
	ST.PhysicalFreq: "CIRCLE",
	
	# Blender
	ST.BlenderObject: "SQUARE",
	ST.BlenderCollection: "SQUARE",
	ST.BlenderImage: "DIAMOND",
	ST.BlenderVolume: "DIAMOND",
	ST.BlenderGeoNodes: "DIAMOND",
	ST.BlenderText: "SQUARE",
	ST.BlenderPreviewTarget: "SQUARE",
	
	# Maxwell
	ST.MaxwellSource: "CIRCLE",
	ST.MaxwellTemporalShape: "CIRCLE",
	ST.MaxwellMedium: "CIRCLE",
	ST.MaxwellMediumNonLinearity: "CIRCLE",
	ST.MaxwellStructure: "SQUARE",
	ST.MaxwellBoundBox: "SQUARE",
	ST.MaxwellBoundFace: "DIAMOND",
	ST.MaxwellMonitor: "CIRCLE",
	ST.MaxwellFDTDSim: "SQUARE",
	ST.MaxwellSimGrid: "SQUARE",
	ST.MaxwellSimGridAxis: "DIAMOND",
	ST.MaxwellSimDomain: "SQUARE",
	
	# Tidy3D
	ST.Tidy3DCloudTask: "CIRCLE",
}
