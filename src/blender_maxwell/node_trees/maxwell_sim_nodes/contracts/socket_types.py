import enum

from blender_maxwell.utils import blender_type_enum


@blender_type_enum.append_cls_name_to_values
class SocketType(blender_type_enum.BlenderTypeEnum):
	Expr = enum.auto()

	# Base
	Any = enum.auto()
	Bool = enum.auto()
	String = enum.auto()
	FilePath = enum.auto()
	Color = enum.auto()

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
	MaxwellMonitorData = enum.auto()

	MaxwellFDTDSim = enum.auto()
	MaxwellFDTDSimData = enum.auto()
	MaxwellSimDomain = enum.auto()
	MaxwellSimGrid = enum.auto()
	MaxwellSimGridAxis = enum.auto()

	# Tidy3D
	Tidy3DCloudTask = enum.auto()

	# Physical
	PhysicalUnitSystem = enum.auto()

	## Optical
	PhysicalPol = enum.auto()
