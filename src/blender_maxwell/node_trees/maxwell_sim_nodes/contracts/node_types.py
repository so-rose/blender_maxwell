import enum

from blender_maxwell.utils import blender_type_enum


@blender_type_enum.append_cls_name_to_values
class NodeType(blender_type_enum.BlenderTypeEnum):
	# KitchenSink = enum.auto()

	# Analysis
	ExtractData = enum.auto()
	Viz = enum.auto()
	## Analysis / Math
	MapMath = enum.auto()
	FilterMath = enum.auto()
	ReduceMath = enum.auto()
	OperateMath = enum.auto()
	TransformMath = enum.auto()

	# Inputs
	WaveConstant = enum.auto()
	Scene = enum.auto()
	## Inputs / Constants
	ExprConstant = enum.auto()
	PhysicalConstant = enum.auto()
	NumberConstant = enum.auto()
	VectorConstant = enum.auto()
	ScientificConstant = enum.auto()
	UnitSystemConstant = enum.auto()
	BlenderConstant = enum.auto()
	## Inputs / Web Importers
	Tidy3DWebImporter = enum.auto()
	## Inputs / File Importers
	DataFileImporter = enum.auto()
	Tidy3DFileImporter = enum.auto()

	# Outputs
	Viewer = enum.auto()
	## Outputs / File Exporters
	Tidy3DWebExporter = enum.auto()
	## Outputs / Web Exporters
	JSONFileExporter = enum.auto()

	# Sources
	## Sources / Temporal Shapes
	GaussianPulseTemporalShape = enum.auto()
	ContinuousWaveTemporalShape = enum.auto()
	SymbolicTemporalShape = enum.auto()
	DataTemporalShape = enum.auto()
	## Sources /
	PointDipoleSource = enum.auto()
	PlaneWaveSource = enum.auto()
	UniformCurrentSource = enum.auto()
	TFSFSource = enum.auto()
	GaussianBeamSource = enum.auto()
	AstigmaticGaussianBeamSource = enum.auto()
	EHDataSource = enum.auto()
	EHEquivDataSource = enum.auto()

	# Mediums
	LibraryMedium = enum.auto()
	DataFitMedium = enum.auto()
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
	BLObjectStructure = enum.auto()
	GeoNodesStructure = enum.auto()
	## Structures / Primitives
	LineStructure = enum.auto()
	PlaneStructure = enum.auto()
	BoxStructure = enum.auto()
	SphereStructure = enum.auto()
	CylinderStructure = enum.auto()
	PolySlabStructure = enum.auto()

	# Bounds
	BoundConds = enum.auto()
	## Bounds / Bound Conds
	PMLBoundCond = enum.auto()
	BlochBoundCond = enum.auto()
	AdiabAbsorbBoundCond = enum.auto()

	# Monitors
	EHFieldMonitor = enum.auto()
	PowerFluxMonitor = enum.auto()
	PermittivityMonitor = enum.auto()
	DiffractionMonitor = enum.auto()
	## Monitors / Projected
	CartesianNearFieldProjectionMonitor = enum.auto()
	AngleNearFieldProjectionMonitor = enum.auto()
	KSpaceNearFieldProjectionMonitor = enum.auto()

	# Sims
	FDTDSim = enum.auto()
	SimDomain = enum.auto()
	SimGrid = enum.auto()
	## Sims / Sim Grid Axis
	AutomaticSimGridAxis = enum.auto()
	ManualSimGridAxis = enum.auto()
	UniformSimGridAxis = enum.auto()
	ArraySimGridAxis = enum.auto()

	# Utilities
	Combine = enum.auto()
	Separate = enum.auto()
