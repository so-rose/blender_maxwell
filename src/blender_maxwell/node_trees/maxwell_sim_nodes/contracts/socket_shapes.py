from .socket_types import SocketType as ST

SOCKET_SHAPES = {
	# Basic
	ST.Any: "CIRCLE",
	ST.Bool: "CIRCLE",
	ST.String: "CIRCLE",
	ST.FilePath: "CIRCLE",
	
	# Number
	ST.IntegerNumber: "CIRCLE",
	ST.RationalNumber: "CIRCLE",
	ST.RealNumber: "CIRCLE",
	ST.ComplexNumber: "CIRCLE",
	
	# Vector
	ST.Integer2DVector: "CIRCLE",
	ST.Real2DVector: "CIRCLE",
	ST.Complex2DVector: "CIRCLE",
	ST.Integer3DVector: "CIRCLE",
	ST.Real3DVector: "CIRCLE",
	ST.Complex3DVector: "CIRCLE",
	
	# Physical
	ST.PhysicalUnitSystem: "CIRCLE",
	ST.PhysicalTime: "CIRCLE",
	ST.PhysicalAngle: "CIRCLE",
	ST.PhysicalLength: "CIRCLE",
	ST.PhysicalArea: "CIRCLE",
	ST.PhysicalVolume: "CIRCLE",
	ST.PhysicalPoint2D: "CIRCLE",
	ST.PhysicalPoint3D: "CIRCLE",
	ST.PhysicalSize2D: "CIRCLE",
	ST.PhysicalSize3D: "CIRCLE",
	ST.PhysicalMass: "CIRCLE",
	ST.PhysicalSpeed: "CIRCLE",
	ST.PhysicalAccelScalar: "CIRCLE",
	ST.PhysicalForceScalar: "CIRCLE",
	ST.PhysicalAccel3D: "CIRCLE",
	ST.PhysicalForce3D: "CIRCLE",
	ST.PhysicalPol: "CIRCLE",
	ST.PhysicalFreq: "CIRCLE",
	
	# Blender
	ST.BlenderObject: "DIAMOND",
	ST.BlenderCollection: "DIAMOND",
	ST.BlenderImage: "DIAMOND",
	ST.BlenderGeoNodes: "DIAMOND",
	ST.BlenderText: "DIAMOND",
	
	# Maxwell
	ST.MaxwellSource: "CIRCLE",
	ST.MaxwellTemporalShape: "CIRCLE",
	ST.MaxwellMedium: "CIRCLE",
	ST.MaxwellMediumNonLinearity: "CIRCLE",
	ST.MaxwellStructure: "CIRCLE",
	ST.MaxwellBoundConds: "CIRCLE",
	ST.MaxwellBoundCond: "CIRCLE",
	ST.MaxwellMonitor: "CIRCLE",
	ST.MaxwellFDTDSim: "CIRCLE",
	ST.MaxwellFDTDSimData: "CIRCLE",
	ST.MaxwellSimGrid: "CIRCLE",
	ST.MaxwellSimGridAxis: "CIRCLE",
	ST.MaxwellSimDomain: "CIRCLE",
	
	# Tidy3D
	ST.Tidy3DCloudTask: "DIAMOND",
}