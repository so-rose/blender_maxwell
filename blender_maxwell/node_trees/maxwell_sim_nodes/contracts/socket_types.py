import enum

from ....utils.blender_type_enum import (
	BlenderTypeEnum, append_cls_name_to_values, wrap_values_in_MT
)

@append_cls_name_to_values
class SocketType(BlenderTypeEnum):
	# Base
	Any = enum.auto()
	Bool = enum.auto()
	Text = enum.auto()
	FilePath = enum.auto()
	Secret = enum.auto()
	
	# Number
	IntegerNumber = enum.auto()
	RationalNumber = enum.auto()
	RealNumber = enum.auto()
	ComplexNumber = enum.auto()
	
	# Vector
	Real2DVector = enum.auto()
	Complex2DVector = enum.auto()
	
	Real3DVector = enum.auto()
	Complex3DVector = enum.auto()
	
	# Physical
	PhysicalUnitSystem = enum.auto()
	PhysicalTime = enum.auto()
	
	PhysicalAngle = enum.auto()
	
	PhysicalLength = enum.auto()
	PhysicalArea = enum.auto()
	PhysicalVolume = enum.auto()
	
	PhysicalPoint2D = enum.auto()
	PhysicalPoint3D = enum.auto()
	
	PhysicalSize2D = enum.auto()
	PhysicalSize3D = enum.auto()
	
	PhysicalMass = enum.auto()
	
	PhysicalSpeed = enum.auto()
	PhysicalAccelScalar = enum.auto()
	PhysicalForceScalar = enum.auto()
	PhysicalAccel3DVector = enum.auto()
	PhysicalForce3DVector = enum.auto()
	
	PhysicalPol = enum.auto()
	
	PhysicalFreq = enum.auto()
	
	# Blender
	BlenderObject = enum.auto()
	BlenderCollection = enum.auto()
	
	BlenderImage = enum.auto()
	BlenderVolume = enum.auto()
	
	BlenderGeoNodes = enum.auto()
	BlenderText = enum.auto()
	
	BlenderPreviewTarget = enum.auto()
	
	# Maxwell
	MaxwellSource = enum.auto()
	MaxwellTemporalShape = enum.auto()
	
	MaxwellMedium = enum.auto()
	MaxwellMediumNonLinearity = enum.auto()
	
	MaxwellStructure = enum.auto()
	
	MaxwellBoundBox = enum.auto()
	MaxwellBoundFace = enum.auto()
	
	MaxwellMonitor = enum.auto()

	MaxwellFDTDSim = enum.auto()
	MaxwellSimGrid = enum.auto()
	MaxwellSimGridAxis = enum.auto()
	MaxwellSimDomain = enum.auto()
	
	# Tidy3D
	Tidy3DCloudTask = enum.auto()
