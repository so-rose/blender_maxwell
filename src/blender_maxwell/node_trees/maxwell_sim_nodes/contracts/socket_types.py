import enum

from blender_maxwell.utils.blender_type_enum import (
	BlenderTypeEnum,
	append_cls_name_to_values,
)


@append_cls_name_to_values
class SocketType(BlenderTypeEnum):
	# Base
	Any = enum.auto()
	Bool = enum.auto()
	String = enum.auto()
	FilePath = enum.auto()
	Color = enum.auto()
	Expr = enum.auto()

	# Number
	IntegerNumber = enum.auto()
	RationalNumber = enum.auto()
	RealNumber = enum.auto()
	ComplexNumber = enum.auto()

	# Vector
	Integer2DVector = enum.auto()
	Real2DVector = enum.auto()
	Real2DVectorDir = enum.auto()
	Complex2DVector = enum.auto()

	Integer3DVector = enum.auto()
	Real3DVector = enum.auto()
	Real3DVectorDir = enum.auto()
	Complex3DVector = enum.auto()

	# Blender
	BlenderMaterial = enum.auto()
	BlenderObject = enum.auto()
	BlenderCollection = enum.auto()

	BlenderImage = enum.auto()

	BlenderGeoNodes = enum.auto()
	BlenderText = enum.auto()

	# Maxwell
	MaxwellBoundConds = enum.auto()
	MaxwellBoundCond = enum.auto()

	MaxwellMedium = enum.auto()
	MaxwellMediumNonLinearity = enum.auto()

	MaxwellSource = enum.auto()
	MaxwellTemporalShape = enum.auto()

	MaxwellStructure = enum.auto()
	MaxwellMonitor = enum.auto()

	MaxwellFDTDSim = enum.auto()
	MaxwellFDTDSimData = enum.auto()
	MaxwellSimDomain = enum.auto()
	MaxwellSimGrid = enum.auto()
	MaxwellSimGridAxis = enum.auto()

	# Tidy3D
	Tidy3DCloudTask = enum.auto()

	# Physical
	PhysicalUnitSystem = enum.auto()

	PhysicalTime = enum.auto()

	PhysicalAngle = enum.auto()
	PhysicalSolidAngle = enum.auto()

	PhysicalRot2D = enum.auto()
	PhysicalRot3D = enum.auto()

	PhysicalFreq = enum.auto()
	PhysicalAngFreq = enum.auto()

	## Cartesian
	PhysicalLength = enum.auto()
	PhysicalArea = enum.auto()
	PhysicalVolume = enum.auto()

	PhysicalDisp2D = enum.auto()
	PhysicalDisp3D = enum.auto()

	PhysicalPoint1D = enum.auto()
	PhysicalPoint2D = enum.auto()
	PhysicalPoint3D = enum.auto()

	PhysicalSize2D = enum.auto()
	PhysicalSize3D = enum.auto()

	## Mechanical
	PhysicalMass = enum.auto()

	PhysicalSpeed = enum.auto()
	PhysicalVel2D = enum.auto()
	PhysicalVel3D = enum.auto()
	PhysicalAccelScalar = enum.auto()
	PhysicalAccel2D = enum.auto()
	PhysicalAccel3D = enum.auto()
	PhysicalForceScalar = enum.auto()
	PhysicalForce2D = enum.auto()
	PhysicalForce3D = enum.auto()
	PhysicalPressure = enum.auto()

	## Energetic
	PhysicalEnergy = enum.auto()
	PhysicalPower = enum.auto()
	PhysicalTemp = enum.auto()

	## Electrodynamical
	PhysicalCurr = enum.auto()
	PhysicalCurrDens2D = enum.auto()
	PhysicalCurrDens3D = enum.auto()

	PhysicalCharge = enum.auto()
	PhysicalVoltage = enum.auto()
	PhysicalCapacitance = enum.auto()
	PhysicalResistance = enum.auto()
	PhysicalConductance = enum.auto()

	PhysicalMagFlux = enum.auto()
	PhysicalMagFluxDens = enum.auto()
	PhysicalInductance = enum.auto()

	PhysicalEField2D = enum.auto()
	PhysicalEField3D = enum.auto()
	PhysicalHField2D = enum.auto()
	PhysicalHField3D = enum.auto()

	## Luminal
	PhysicalLumIntensity = enum.auto()
	PhysicalLumFlux = enum.auto()
	PhysicalIlluminance = enum.auto()

	## Optical
	PhysicalPolJones = enum.auto()
	PhysicalPol = enum.auto()
