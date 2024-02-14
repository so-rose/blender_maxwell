import typing as typ
import typing_extensions as pytypes_ext
import enum

import sympy as sp
import sympy.physics.units as spu
import pydantic as pyd
import bpy

from ...utils.blender_type_enum import (
	BlenderTypeEnum, append_cls_name_to_values
)

####################
# - String Types
####################
BlenderColorRGB = tuple[float, float, float, float]
BlenderID = pytypes_ext.Annotated[str, pyd.StringConstraints(
	pattern=r'^[A-Z_]+$',
)]

# Socket ID
SocketName = pytypes_ext.Annotated[str, pyd.StringConstraints(
	pattern=r'^[a-zA-Z0-9_]+$',
)]
BLSocketName = pytypes_ext.Annotated[str, pyd.StringConstraints(
	pattern=r'^[a-zA-Z0-9_]+$',
)]

# Socket ID
PresetID = pytypes_ext.Annotated[str, pyd.StringConstraints(
	pattern=r'^[A-Z_]+$',
)]

####################
# - Generic Types
####################
SocketReturnType = typ.TypeVar('SocketReturnType', covariant=True)
## - Covariance: If B subtypes A, then Container[B] subtypes Container[A].
## - This is absolutely what we want here.

####################
# - Sympy Expression Typing
####################
ALL_UNIT_SYMBOLS = {
	unit
	for unit in spu.__dict__.values()
	if isinstance(unit, spu.Quantity)
}
def has_units(expr: sp.Expr):
	return any(
		symbol in ALL_UNIT_SYMBOLS
		for symbol in expr.atoms(sp.Symbol)
	)
def is_exactly_expressed_as_unit(expr: sp.Expr, unit) -> bool:
	#try:
	converted_expr = expr / unit
	
	return (
		converted_expr.is_number
		and not converted_expr.has(spu.Quantity)
	)

####################
# - Icon Types
####################
class Icon(BlenderTypeEnum):
	MaxwellSimTree = "MOD_SIMPLEDEFORM"

####################
# - Tree Types
####################
@append_cls_name_to_values
class TreeType(BlenderTypeEnum):
	MaxwellSim = enum.auto()

####################
# - Socket Types
####################
@append_cls_name_to_values
class SocketType(BlenderTypeEnum):
	Any = enum.auto()
	Text = enum.auto()
	FilePath = enum.auto()
	
	RationalNumber = enum.auto()
	RealNumber = enum.auto()
	ComplexNumber = enum.auto()
	
	PhysicalLength = enum.auto()
	PhysicalArea = enum.auto()
	
	MaxwellSource = enum.auto()
	MaxwellMedium = enum.auto()
	MaxwellStructure = enum.auto()
	MaxwellBound = enum.auto()
	MaxwellFDTDSim = enum.auto()

####################
# - Node Types
####################
@append_cls_name_to_values
class NodeType(BlenderTypeEnum):
	# Inputs
	Time = enum.auto()
	ObjectInfo = enum.auto()
	
	FloatParameter = enum.auto()
	ComplexParameter = enum.auto()
	Vec3Parameter = enum.auto()
	
	ScientificConstant = enum.auto()
	FloatConstant = enum.auto()
	ComplexConstant = enum.auto()
	Vec3Constant = enum.auto()
	
	FloatArrayElement = enum.auto()
	ComplexArrayElement = enum.auto()
	Vec3ArrayElement = enum.auto()
	
	FloatDictElement = enum.auto()
	ComplexDictElement = enum.auto()
	Vec3DictElement = enum.auto()
	
	FloatField = enum.auto()
	ComplexField = enum.auto()
	Vec3Field = enum.auto()
	
	# Outputs
	ValueViewer = enum.auto()
	ConsoleViewer = enum.auto()
	
	JSONFileExporter = enum.auto()
	
	# Viz
	TemporalShapeViz = enum.auto()
	SourceViz = enum.auto()
	StructureViz = enum.auto()
	BoundViz = enum.auto()
	FDTDViz = enum.auto()
	
	# Sources
	GaussianPulseTemporalShape = enum.auto()
	ContinuousWaveTemporalShape = enum.auto()
	DataDrivenTemporalShape = enum.auto()
	
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
	
	AddNonLinearity = enum.auto()
	ChiThreeSusceptibilityNonLinearity = enum.auto()
	TwoPhotonAbsorptionNonLinearity = enum.auto()
	KerrNonLinearity = enum.auto()
	
	# Structures
	TriMeshStructure = enum.auto()
	
	BoxStructure = enum.auto()
	SphereStructure = enum.auto()
	CylinderStructure = enum.auto()
	
	GeoNodesStructure = enum.auto()
	ScriptedStructure = enum.auto()
	
	# Bounds
	BoundBox = enum.auto()
	
	PMLBoundFace = enum.auto()
	PECBoundFace = enum.auto()
	
	BlochBoundFace = enum.auto()
	PeriodicBoundFace = enum.auto()
	AbsorbingBoundFace = enum.auto()
	
	# Monitors
	EHFieldMonitor = enum.auto()
	FieldPowerFluxMonitor = enum.auto()
	EpsilonTensorMonitor = enum.auto()
	DiffractionMonitor = enum.auto()
	
	CartesianNearFieldProjectionMonitor = enum.auto()
	ObservationAngleNearFieldProjectionMonitor = enum.auto()
	KSpaceNearFieldProjectionMonitor = enum.auto()
	
	# Simulations
	FDTDSim = enum.auto()
	
	SimulationGridDiscretization = enum.auto()
	
	Automatic1DGridDiscretization = enum.auto()
	Manual1DGridDiscretization = enum.auto()
	Uniform1DGridDiscretization = enum.auto()
	DataDriven1DGridDiscretization = enum.auto()
	
	# Utilities
	FloatMath = enum.auto()
	ComplexMath = enum.auto()
	Vec3Math = enum.auto()
	
	FloatFieldMath = enum.auto()
	ComplexFieldMath = enum.auto()
	Vec3FieldMath = enum.auto()
	
	SpectralMath = enum.auto()

####################
# - Node Category Types
####################
class NodeCategory(BlenderTypeEnum):
	MAXWELL_SIM = enum.auto()
	
	# Inputs/
	MAXWELL_SIM_INPUTS = enum.auto()
	MAXWELL_SIM_INPUTS_SCENE = enum.auto()
	MAXWELL_SIM_INPUTS_PARAMETERS = enum.auto()
	MAXWELL_SIM_INPUTS_CONSTANTS = enum.auto()
	MAXWELL_SIM_INPUTS_ARRAY = enum.auto()
	MAXWELL_SIM_INPUTS_ARRAY_ELEMENTS = enum.auto()
	MAXWELL_SIM_INPUTS_ARRAY_UNIONS = enum.auto()
	MAXWELL_SIM_INPUTS_DICTIONARY = enum.auto()
	MAXWELL_SIM_INPUTS_DICTIONARY_ELEMENTS = enum.auto()
	MAXWELL_SIM_INPUTS_DICTIONARY_UNIONS = enum.auto()
	MAXWELL_SIM_INPUTS_FIELDS = enum.auto()
	
	# Outputs/
	MAXWELL_SIM_OUTPUTS = enum.auto()
	MAXWELL_SIM_OUTPUTS_VIEWERS = enum.auto()
	MAXWELL_SIM_OUTPUTS_EXPORTERS = enum.auto()
	
	# Viz/
	MAXWELL_SIM_VIZ = enum.auto()
	
	# Sources/
	MAXWELL_SIM_SOURCES = enum.auto()
	MAXWELL_SIM_SOURCES_TEMPORALSHAPES = enum.auto()
	MAXWELL_SIM_SOURCES_MODELLED = enum.auto()
	MAXWELL_SIM_SOURCES_DATADRIVEN = enum.auto()
	
	# Mediums/
	MAXWELL_SIM_MEDIUMS = enum.auto()
	MAXWELL_SIM_MEDIUMS_LINEARMEDIUMS = enum.auto()
	MAXWELL_SIM_MEDIUMS_LINEARMEDIUMS_DIRECT = enum.auto()
	MAXWELL_SIM_MEDIUMS_LINEARMEDIUMS_MODELLED = enum.auto()
	MAXWELL_SIM_MEDIUMS_NONLINEARITIES = enum.auto()
	
	# Structures/
	MAXWELL_SIM_STRUCTURES = enum.auto()
	MAXWELL_SIM_STRUCTURES_PRIMITIES = enum.auto()
	MAXWELL_SIM_STRUCTURES_GENERATED = enum.auto()
	
	# Bounds/
	MAXWELL_SIM_BOUNDS = enum.auto()
	MAXWELL_SIM_BOUNDS_BOUNDFACES = enum.auto()
	
	# Monitors/
	MAXWELL_SIM_MONITORS = enum.auto()
	MAXWELL_SIM_MONITORS_NEARFIELDPROJECTIONS = enum.auto()
	
	# Simulations/
	MAXWELL_SIM_SIMULATIONS = enum.auto()
	MAXWELL_SIM_SIMULATIONS_DISCRETIZATIONS = enum.auto()
	MAXWELL_SIM_SIMULATIONS_DISCRETIZATIONS_1DGRID = enum.auto()
	
	# Utilities/
	MAXWELL_SIM_UTILITIES = enum.auto()
	MAXWELL_SIM_UTILITIES_MATH = enum.auto()
	MAXWELL_SIM_UTILITIES_FIELDMATH = enum.auto()
	
	@classmethod
	def get_tree(cls):
		## TODO: Refactor
		syllable_categories = [
			node_category.value.split("_")
			for node_category in cls
			if node_category.value != "MAXWELL_SIM"
		]
		
		category_tree = {}
		for syllable_category in syllable_categories:
			# Set Current Subtree to Root
			current_category_subtree = category_tree
			
			for i, syllable in enumerate(syllable_category):
				# Create New Category Subtree and/or Step to Subtree
				if syllable not in current_category_subtree:
					current_category_subtree[syllable] = {}
				current_category_subtree = current_category_subtree[syllable]
		
		return category_tree

NodeCategory_to_category_label = {
	# Inputs/
	NodeCategory.MAXWELL_SIM_INPUTS: "Inputs",
	NodeCategory.MAXWELL_SIM_INPUTS_SCENE: "Scene",
	NodeCategory.MAXWELL_SIM_INPUTS_PARAMETERS: "Parameters",
	NodeCategory.MAXWELL_SIM_INPUTS_CONSTANTS: "Constants",
	NodeCategory.MAXWELL_SIM_INPUTS_ARRAY: "Array",
	NodeCategory.MAXWELL_SIM_INPUTS_ARRAY_ELEMENTS: "Elements",
	NodeCategory.MAXWELL_SIM_INPUTS_ARRAY_UNIONS: "Unions",
	NodeCategory.MAXWELL_SIM_INPUTS_DICTIONARY: "Dictionary",
	NodeCategory.MAXWELL_SIM_INPUTS_DICTIONARY_ELEMENTS: "Elements",
	NodeCategory.MAXWELL_SIM_INPUTS_DICTIONARY_UNIONS: "Unions",
	NodeCategory.MAXWELL_SIM_INPUTS_FIELDS: "Fields",
	
	# Outputs/
	NodeCategory.MAXWELL_SIM_OUTPUTS: "Outputs",
	NodeCategory.MAXWELL_SIM_OUTPUTS_VIEWERS: "Viewers",
	NodeCategory.MAXWELL_SIM_OUTPUTS_EXPORTERS: "Exporters",
	
	# Viz/
	NodeCategory.MAXWELL_SIM_VIZ: "Viz",
	
	# Sources/
	NodeCategory.MAXWELL_SIM_SOURCES: "Sources",
	NodeCategory.MAXWELL_SIM_SOURCES_TEMPORALSHAPES: "Temporal Shapes",
	NodeCategory.MAXWELL_SIM_SOURCES_MODELLED: "Modelled",
	NodeCategory.MAXWELL_SIM_SOURCES_DATADRIVEN: "Data-Driven",
	
	# Mediums/
	NodeCategory.MAXWELL_SIM_MEDIUMS: "Mediums",
	NodeCategory.MAXWELL_SIM_MEDIUMS_LINEARMEDIUMS: "Linear Mediums",
	NodeCategory.MAXWELL_SIM_MEDIUMS_LINEARMEDIUMS_DIRECT: "Direct",
	NodeCategory.MAXWELL_SIM_MEDIUMS_LINEARMEDIUMS_MODELLED: "Modelled",
	NodeCategory.MAXWELL_SIM_MEDIUMS_NONLINEARITIES: "Non-Linearities",
	
	# Structures/
	NodeCategory.MAXWELL_SIM_STRUCTURES: "Structures",
	NodeCategory.MAXWELL_SIM_STRUCTURES_PRIMITIES: "Primitives",
	NodeCategory.MAXWELL_SIM_STRUCTURES_GENERATED: "Generated",
	
	# Bounds/
	NodeCategory.MAXWELL_SIM_BOUNDS: "Bounds",
	NodeCategory.MAXWELL_SIM_BOUNDS_BOUNDFACES: "Bound Faces",
	
	# Monitors/
	NodeCategory.MAXWELL_SIM_MONITORS: "Monitors",
	NodeCategory.MAXWELL_SIM_MONITORS_NEARFIELDPROJECTIONS: "Near-Field Projections",
	
	# Simulations/
	NodeCategory.MAXWELL_SIM_SIMULATIONS: "Simulations",
	NodeCategory.MAXWELL_SIM_SIMULATIONS_DISCRETIZATIONS: "Discretizations",
	NodeCategory.MAXWELL_SIM_SIMULATIONS_DISCRETIZATIONS_1DGRID: "1D Grid Discretizations",
	
	# Utilities/
	NodeCategory.MAXWELL_SIM_UTILITIES: "Utilities",
	NodeCategory.MAXWELL_SIM_UTILITIES_MATH: "Math",
	NodeCategory.MAXWELL_SIM_UTILITIES_FIELDMATH: "Field Math",
}



####################
# - Protocols
####################
class SocketDefProtocol(typ.Protocol):
	socket_type: SocketType
	label: str
	
	def init(self, bl_socket: bpy.types.NodeSocket) -> None:
		...

class PresetDef(pyd.BaseModel):
	label: str
	description: str
	values: dict[SocketName, typ.Any]

@typ.runtime_checkable
#class BLSocketProtocol(typ.Protocol):
#	socket_type: SocketType
#	socket_color: BlenderColorRGB
#	
#	bl_label: str
#	
#	compatible_types: dict[typ.Type, set[typ.Callable[[typ.Any], bool]]]
#	
#	def draw(
#		self,
#		context: bpy.types.Context,
#		layout: bpy.types.UILayout,
#		node: bpy.types.Node,
#		text: str,
#	) -> None:
#		...
#	
#	@property
#	def default_value(self) -> typ.Any:
#		...
#	@default_value.setter
#	def default_value(self, value: typ.Any) -> typ.Any:
#		...
#	

@typ.runtime_checkable
class NodeTypeProtocol(typ.Protocol):
	node_type: NodeType
	
	bl_label: str
	
	input_sockets: dict[SocketName, SocketDefProtocol]
	output_sockets: dict[SocketName, SocketDefProtocol]
	presets: dict[PresetID, PresetDef] | None
	
	# Built-In Blender Methods
	def init(self, context: bpy.types.Context) -> None:
		...
	
	def draw_buttons(
		self,
		context: bpy.types.Context,
		layout: bpy.types.UILayout,
	) -> None:
		...
	
	@classmethod
	def poll(cls, ntree: bpy.types.NodeTree) -> None:
		...
	
	# Socket Getters
	def g_input_bl_socket(
		self,
		input_socket_name: SocketName,
	) -> bpy.types.NodeSocket:
		...
	
	def g_output_bl_socket(
		self,
		output_socket_name: SocketName,
	) -> bpy.types.NodeSocket:
		...
	
	# Socket Methods
	def s_input_value(
		self,
		input_socket_name: SocketName,
		value: typ.Any
	) -> typ.Any:
		...
	
	# Data-Flow Methods
	def compute_input(
		self,
		input_socket_name: SocketName,
	) -> typ.Any:
		...
	def compute_output(
		self,
		output_socket_name: SocketName,
	) -> typ.Any:
		...
