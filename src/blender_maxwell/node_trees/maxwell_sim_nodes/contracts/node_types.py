import enum

from blender_maxwell.utils import blender_type_enum


@blender_type_enum.append_cls_name_to_values
class NodeType(blender_type_enum.BlenderTypeEnum):
	# KitchenSink = enum.auto()

	# Analysis
	Viz = enum.auto()
	ExtractData = enum.auto()
	## Analysis / Math
	MapMath = enum.auto()
	FilterMath = enum.auto()
	ReduceMath = enum.auto()
	OperateMath = enum.auto()
	TransformMath = enum.auto()

	# Inputs
	WaveConstant = enum.auto()
	UnitSystem = enum.auto()

	## Inputs / Scene
	# Time = enum.auto()
	## Inputs / Web Importers
	Tidy3DWebImporter = enum.auto()
	## Inputs / File Importers
	Tidy3DFileImporter = enum.auto()
	## Inputs / Constants
	ExprConstant = enum.auto()
	ScientificConstant = enum.auto()
	NumberConstant = enum.auto()
	PhysicalConstant = enum.auto()
	BlenderConstant = enum.auto()

	# Outputs
	Viewer = enum.auto()
	## Outputs / File Exporters
	Tidy3DWebExporter = enum.auto()
	## Outputs / Web Exporters
	JSONFileExporter = enum.auto()

	# Sources
	## Sources /
	PointDipoleSource = enum.auto()
	PlaneWaveSource = enum.auto()
	UniformCurrentSource = enum.auto()
	# ModeSource = enum.auto()
	# GaussianBeamSource = enum.auto()
	# AstigmaticGaussianBeamSource = enum.auto()
	# TFSFSource = enum.auto()
	# EHEquivalenceSource = enum.auto()
	# EHSource = enum.auto()
	## Sources / Temporal Shapes
	GaussianPulseTemporalShape = enum.auto()
	# ContinuousWaveTemporalShape = enum.auto()
	# ArrayTemporalShape = enum.auto()

	# Mediums
	LibraryMedium = enum.auto()
	# PECMedium = enum.auto()
	# IsotropicMedium = enum.auto()
	# AnisotropicMedium = enum.auto()
	# TripleSellmeierMedium = enum.auto()
	# SellmeierMedium = enum.auto()
	# PoleResidueMedium = enum.auto()
	# DrudeMedium = enum.auto()
	# DrudeLorentzMedium = enum.auto()
	# DebyeMedium = enum.auto()

	## Mediums / Non-Linearities
	# AddNonLinearity = enum.auto()
	# ChiThreeSusceptibilityNonLinearity = enum.auto()
	# TwoPhotonAbsorptionNonLinearity = enum.auto()
	# KerrNonLinearity = enum.auto()

	# Structures
	# ObjectStructure = enum.auto()
	GeoNodesStructure = enum.auto()
	# ScriptedStructure = enum.auto()
	## Structures / Primitives
	BoxStructure = enum.auto()
	SphereStructure = enum.auto()
	# CylinderStructure = enum.auto()

	# Bounds
	BoundConds = enum.auto()
	## Bounds / Bound Conds
	PMLBoundCond = enum.auto()
	PECBoundCond = enum.auto()
	PMCBoundCond = enum.auto()
	BlochBoundCond = enum.auto()
	PeriodicBoundCond = enum.auto()
	AbsorbingBoundCond = enum.auto()

	# Monitors
	EHFieldMonitor = enum.auto()
	PowerFluxMonitor = enum.auto()
	# EpsilonTensorMonitor = enum.auto()
	# DiffractionMonitor = enum.auto()
	## Monitors / Projected
	# CartesianNearFieldProjectionMonitor = enum.auto()
	# ObservationAngleNearFieldProjectionMonitor = enum.auto()
	# KSpaceNearFieldProjectionMonitor = enum.auto()

	# Sims
	FDTDSim = enum.auto()
	SimDomain = enum.auto()
	SimGrid = enum.auto()
	## Sims / Sim Grid Axis
	# AutomaticSimGridAxis = enum.auto()
	# ManualSimGridAxis = enum.auto()
	# UniformSimGridAxis = enum.auto()
	# ArraySimGridAxis = enum.auto()

	# Utilities
	Combine = enum.auto()
	Separate = enum.auto()
