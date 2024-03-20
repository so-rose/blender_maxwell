import enum

from ....utils.blender_type_enum import (
	BlenderTypeEnum,
	append_cls_name_to_values,
)


@append_cls_name_to_values
class NodeType(BlenderTypeEnum):
	KitchenSink = enum.auto()

	# Inputs
	UnitSystem = enum.auto()

	## Inputs / Scene
	Time = enum.auto()

	## Inputs / Importers
	Tidy3DWebImporter = enum.auto()

	## Inputs / Parameters
	NumberParameter = enum.auto()
	PhysicalParameter = enum.auto()

	## Inputs / Constants
	WaveConstant = enum.auto()
	ScientificConstant = enum.auto()
	NumberConstant = enum.auto()
	PhysicalConstant = enum.auto()
	BlenderConstant = enum.auto()

	## Inputs / Lists
	RealList = enum.auto()
	ComplexList = enum.auto()

	## Inputs /
	InputFile = enum.auto()

	# Outputs
	## Outputs / Viewers
	Viewer = enum.auto()
	ValueViewer = enum.auto()
	ConsoleViewer = enum.auto()

	## Outputs / Exporters
	JSONFileExporter = enum.auto()
	Tidy3DWebExporter = enum.auto()

	# Sources
	## Sources / Temporal Shapes
	GaussianPulseTemporalShape = enum.auto()
	ContinuousWaveTemporalShape = enum.auto()
	ListTemporalShape = enum.auto()

	## Sources /
	PointDipoleSource = enum.auto()
	UniformCurrentSource = enum.auto()
	PlaneWaveSource = enum.auto()
	ModeSource = enum.auto()
	GaussianBeamSource = enum.auto()
	AstigmaticGaussianBeamSource = enum.auto()
	TFSFSource = enum.auto()

	EHEquivalenceSource = enum.auto()
	EHSource = enum.auto()

	# Mediums
	LibraryMedium = enum.auto()

	PECMedium = enum.auto()
	IsotropicMedium = enum.auto()
	AnisotropicMedium = enum.auto()

	TripleSellmeierMedium = enum.auto()
	SellmeierMedium = enum.auto()
	PoleResidueMedium = enum.auto()
	DrudeMedium = enum.auto()
	DrudeLorentzMedium = enum.auto()
	DebyeMedium = enum.auto()

	## Mediums / Non-Linearities
	AddNonLinearity = enum.auto()
	ChiThreeSusceptibilityNonLinearity = enum.auto()
	TwoPhotonAbsorptionNonLinearity = enum.auto()
	KerrNonLinearity = enum.auto()

	# Structures
	ObjectStructure = enum.auto()
	GeoNodesStructure = enum.auto()
	ScriptedStructure = enum.auto()

	## Structures / Primitives
	BoxStructure = enum.auto()
	SphereStructure = enum.auto()
	CylinderStructure = enum.auto()

	# Bounds
	BoundConds = enum.auto()

	## Bounds / Bound Faces
	PMLBoundCond = enum.auto()
	PECBoundCond = enum.auto()
	PMCBoundCond = enum.auto()

	BlochBoundCond = enum.auto()
	PeriodicBoundCond = enum.auto()
	AbsorbingBoundCond = enum.auto()

	# Monitors
	EHFieldMonitor = enum.auto()
	FieldPowerFluxMonitor = enum.auto()
	EpsilonTensorMonitor = enum.auto()
	DiffractionMonitor = enum.auto()

	## Monitors / Near-Field Projections
	CartesianNearFieldProjectionMonitor = enum.auto()
	ObservationAngleNearFieldProjectionMonitor = enum.auto()
	KSpaceNearFieldProjectionMonitor = enum.auto()

	# Sims
	SimDomain = enum.auto()
	SimGrid = enum.auto()

	## Sims / Sim Grid Axis
	AutomaticSimGridAxis = enum.auto()
	ManualSimGridAxis = enum.auto()
	UniformSimGridAxis = enum.auto()
	ArraySimGridAxis = enum.auto()

	## Sim /
	FDTDSim = enum.auto()

	# Utilities
	Combine = enum.auto()
	Separate = enum.auto()
	Math = enum.auto()

	## Utilities / Converters
	WaveConverter = enum.auto()

	## Utilities / Operations
	ArrayOperation = enum.auto()

	# Viz
	FDTDSimDataViz = enum.auto()
