import typing as typ
import typing_extensions as pytypes_ext
import enum

import sympy as sp

sp.printing.str.StrPrinter._default_settings['abbrev'] = True
## When we str() a unit expression, use abbrevied units.

import sympy.physics.units as spu
import pydantic as pyd
import bpy

from ...utils.blender_type_enum import (
	BlenderTypeEnum, append_cls_name_to_values, wrap_values_in_MT
)
from ...utils import extra_sympy_units as spuex

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
	# Base
	Any = enum.auto()
	Text = enum.auto()
	FilePath = enum.auto()
	
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
	PhysicalTime = enum.auto()
	
	PhysicalAngle = enum.auto()
	
	PhysicalLength = enum.auto()
	PhysicalArea = enum.auto()
	PhysicalVolume = enum.auto()
	
	PhysicalMass = enum.auto()
	
	PhysicalSpeed = enum.auto()
	PhysicalAccel = enum.auto()
	PhysicalForce = enum.auto()
	
	PhysicalPol = enum.auto()
	
	PhysicalFreq = enum.auto()
	PhysicalSpecPowerDist = enum.auto()
	PhysicalSpecRelPermDist = enum.auto()
	
	# Blender
	BlenderObject = enum.auto()
	BlenderCollection = enum.auto()
	
	BlenderImage = enum.auto()
	BlenderVolume = enum.auto()
	
	BlenderGeoNodes = enum.auto()
	BlenderText = enum.auto()
	
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

SocketType_to_units = {
	SocketType.PhysicalTime: {
		"default": "PS",
		"values": {
			"PS": spu.picosecond,
			"NS": spu.nanosecond,
			"MS": spu.microsecond,
			"MLSEC": spu.millisecond,
			"SEC": spu.second,
			"MIN": spu.minute,
			"HOUR": spu.hour,
			"DAY": spu.day,
		},
	},
	
	SocketType.PhysicalAngle: {
		"default": "RADIAN",
		"values": {
			"RADIAN": spu.radian,
			"DEGREE": spu.degree,
			"STERAD": spu.steradian,
			"ANGMIL": spu.angular_mil,
		},
	},
	
	SocketType.PhysicalLength: {
		"default": "UM",
		"values": {
			"PM": spu.picometer,
			"A": spu.angstrom,
			"NM": spu.nanometer,
			"UM": spu.micrometer,
			"MM": spu.millimeter,
			"CM": spu.centimeter,
			"M": spu.meter,
			"INCH": spu.inch,
			"FOOT": spu.foot,
			"YARD": spu.yard,
			"MILE": spu.mile,
		},
	},
	SocketType.PhysicalArea: {
		"default": "UM_SQ",
		"values": {
			"PM_SQ": spu.picometer**2,
			"A_SQ": spu.angstrom**2,
			"NM_SQ": spu.nanometer**2,
			"UM_SQ": spu.micrometer**2,
			"MM_SQ": spu.millimeter**2,
			"CM_SQ": spu.centimeter**2,
			"M_SQ": spu.meter**2,
			"INCH_SQ": spu.inch**2,
			"FOOT_SQ": spu.foot**2,
			"YARD_SQ": spu.yard**2,
			"MILE_SQ": spu.mile**2,
		},
	},
	SocketType.PhysicalVolume: {
		"default": "UM_CB",
		"values": {
			"PM_CB": spu.picometer**3,
			"A_CB": spu.angstrom**3,
			"NM_CB": spu.nanometer**3,
			"UM_CB": spu.micrometer**3,
			"MM_CB": spu.millimeter**3,
			"CM_CB": spu.centimeter**3,
			"M_CB": spu.meter**3,
			"ML": spu.milliliter,
			"L": spu.liter,
			"INCH_CB": spu.inch**3,
			"FOOT_CB": spu.foot**3,
			"YARD_CB": spu.yard**3,
			"MILE_CB": spu.mile**3,
		},
	},
	
	SocketType.PhysicalMass: {
		"default": "UG",
		"values": {
			"E_REST": spu.electron_rest_mass,
			"DAL": spu.dalton,
			"UG": spu.microgram,
			"MG": spu.milligram,
			"G": spu.gram,
			"KG": spu.kilogram,
			"TON": spu.metric_ton,
		},
	},
	
	SocketType.PhysicalSpeed: {
		"default": "UM_S",
		"values": {
			"PM_S": spu.picometer / spu.second,
			"NM_S": spu.nanometer / spu.second,
			"UM_S": spu.micrometer / spu.second,
			"MM_S": spu.millimeter / spu.second,
			"M_S": spu.meter / spu.second,
			"KM_S": spu.kilometer / spu.second,
			"KM_H": spu.kilometer / spu.hour,
			"FT_S": spu.feet / spu.second,
			"MI_H": spu.mile / spu.hour,
		},
	},
	SocketType.PhysicalAccel: {
		"default": "UM_S_SQ",
		"values": {
			"PM_S_SQ": spu.picometer / spu.second**2,
			"NM_S_SQ": spu.nanometer / spu.second**2,
			"UM_S_SQ": spu.micrometer / spu.second**2,
			"MM_S_SQ": spu.millimeter / spu.second**2,
			"M_S_SQ": spu.meter / spu.second**2,
			"KM_S_SQ": spu.kilometer / spu.second**2,
			"FT_S_SQ": spu.feet / spu.second**2,
		},
	},
	SocketType.PhysicalForce: {
		"default": "UNEWT",
		"values": {
			"KG_M_S_SQ": spu.kg * spu.m/spu.second**2,
			"NNEWT": spuex.nanonewton,
			"UNEWT": spuex.micronewton,
			"MNEWT": spuex.millinewton,
			"NEWT": spu.newton,
		},
	},
	
	SocketType.PhysicalFreq: {
		"default": "THZ",
		"values": {
			"HZ": spu.hertz,
			"KHZ": spuex.kilohertz,
			"MHZ": spuex.megahertz,
			"GHZ": spuex.gigahertz,
			"THZ": spuex.terahertz,
			"PHZ": spuex.petahertz,
			#"EHZ": spu.exahertz,
			"VAC_PM": spu.picometer,  ## c(vac) = wl*freq
			"VAC_A": spu.angstrom,
			"VAC_NM": spu.nanometer,
			"VAC_UM": spu.micrometer,
			"VAC_MM": spu.millimeter,
			"VAC_CM": spu.centimeter,
			"VAC_M": spu.meter,
		},
	},
}

SocketType_to_color = {
	# Basic
	SocketType.Any: (0.8, 0.8, 0.8, 1.0),  # Light Grey
	SocketType.Text: (0.7, 0.7, 0.7, 1.0),  # Medium Light Grey
	SocketType.FilePath: (0.6, 0.6, 0.6, 1.0),  # Medium Grey

	# Number
	SocketType.IntegerNumber: (0.5, 0.5, 1.0, 1.0),  # Light Blue
	SocketType.RationalNumber: (0.4, 0.4, 0.9, 1.0),  # Medium Light Blue
	SocketType.RealNumber: (0.3, 0.3, 0.8, 1.0),  # Medium Blue
	SocketType.ComplexNumber: (0.2, 0.2, 0.7, 1.0),  # Dark Blue

	# Vector
	SocketType.Real2DVector: (0.5, 1.0, 0.5, 1.0),  # Light Green
	SocketType.Complex2DVector: (0.4, 0.9, 0.4, 1.0),  # Medium Light Green
	SocketType.Real3DVector: (0.3, 0.8, 0.3, 1.0),  # Medium Green
	SocketType.Complex3DVector: (0.2, 0.7, 0.2, 1.0),  # Dark Green

	# Physical
	SocketType.PhysicalTime: (1.0, 0.5, 0.5, 1.0),  # Light Red
	SocketType.PhysicalAngle: (0.9, 0.45, 0.45, 1.0),  # Medium Light Red
	SocketType.PhysicalLength: (0.8, 0.4, 0.4, 1.0),  # Medium Red
	SocketType.PhysicalArea: (0.7, 0.35, 0.35, 1.0),  # Medium Dark Red
	SocketType.PhysicalVolume: (0.6, 0.3, 0.3, 1.0),  # Dark Red
	SocketType.PhysicalMass: (0.9, 0.6, 0.4, 1.0),  # Light Orange
	SocketType.PhysicalSpeed: (0.8, 0.55, 0.35, 1.0),  # Medium Light Orange
	SocketType.PhysicalAccel: (0.7, 0.5, 0.3, 1.0),  # Medium Orange
	SocketType.PhysicalForce: (0.6, 0.45, 0.25, 1.0),  # Medium Dark Orange
	SocketType.PhysicalPol: (0.5, 0.4, 0.2, 1.0),  # Dark Orange
	SocketType.PhysicalFreq: (1.0, 0.7, 0.5, 1.0),  # Light Peach
	SocketType.PhysicalSpecPowerDist: (0.9, 0.65, 0.45, 1.0),  # Medium Light Peach
	SocketType.PhysicalSpecRelPermDist: (0.8, 0.6, 0.4, 1.0),  # Medium Peach

	# Blender
	SocketType.BlenderObject: (0.7, 0.5, 1.0, 1.0),  # Light Purple
	SocketType.BlenderCollection: (0.6, 0.45, 0.9, 1.0),  # Medium Light Purple
	SocketType.BlenderImage: (0.5, 0.4, 0.8, 1.0),  # Medium Purple
	SocketType.BlenderVolume: (0.4, 0.35, 0.7, 1.0),  # Medium Dark Purple
	SocketType.BlenderGeoNodes: (0.3, 0.3, 0.6, 1.0),  # Dark Purple
	SocketType.BlenderText: (0.5, 0.5, 0.75, 1.0),  # Light Lavender

	# Maxwell
	SocketType.MaxwellSource: (1.0, 1.0, 0.5, 1.0),  # Light Yellow
	SocketType.MaxwellTemporalShape: (0.9, 0.9, 0.45, 1.0),  # Medium Light Yellow
	SocketType.MaxwellMedium: (0.8, 0.8, 0.4, 1.0),  # Medium Yellow
	SocketType.MaxwellMediumNonLinearity: (0.7, 0.7, 0.35, 1.0),  # Medium Dark Yellow
	SocketType.MaxwellStructure: (0.6, 0.6, 0.3, 1.0),  # Dark Yellow
	SocketType.MaxwellBoundBox: (0.9, 0.8, 0.5, 1.0),  # Light Gold
	SocketType.MaxwellBoundFace: (0.8, 0.7, 0.45, 1.0),  # Medium Light Gold
	SocketType.MaxwellMonitor: (0.7, 0.6, 0.4, 1.0),  # Medium Gold
	SocketType.MaxwellFDTDSim: (0.6, 0.5, 0.35, 1.0),  # Medium Dark Gold
	SocketType.MaxwellSimGrid: (0.5, 0.4, 0.3, 1.0),  # Dark Gold
	SocketType.MaxwellSimGridAxis: (0.4, 0.3, 0.25, 1.0),  # Darkest Gold
}

####################
# - Node Types
####################
@append_cls_name_to_values
class NodeType(BlenderTypeEnum):
	KitchenSink = enum.auto()
	
	# Inputs
	## Inputs / Scene
	Time = enum.auto()
	UnitSystem = enum.auto()
	
	## Inputs / Parameters
	NumberParameter = enum.auto()
	PhysicalParameter = enum.auto()
	
	## Inputs / Constants
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
	ValueViewer = enum.auto()
	ConsoleViewer = enum.auto()
	
	## Outputs / Exporters
	JSONFileExporter = enum.auto()
	
	
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
	BoundBox = enum.auto()
	
	## Bounds / Bound Faces
	PMLBoundFace = enum.auto()
	PECBoundFace = enum.auto()
	PMCBoundFace = enum.auto()
	
	BlochBoundFace = enum.auto()
	PeriodicBoundFace = enum.auto()
	AbsorbingBoundFace = enum.auto()
	
	
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
	SimGrid = enum.auto()
	
	## Sims / Sim Grid Axis
	AutomaticSimGridAxis = enum.auto()
	ManualSimGridAxis = enum.auto()
	UniformSimGridAxis = enum.auto()
	ArraySimGridAxis = enum.auto()
	
	## Sim /
	FDTDSim = enum.auto()
	
	
	# Utilities
	Math = enum.auto()
	
	## Utilities / Operations
	ArrayOperation = enum.auto()

####################
# - Node Category Types
####################
@wrap_values_in_MT
class NodeCategory(BlenderTypeEnum):
	MAXWELLSIM = enum.auto()
	
	# Inputs/
	MAXWELLSIM_INPUTS = enum.auto()
	MAXWELLSIM_INPUTS_SCENE = enum.auto()
	MAXWELLSIM_INPUTS_PARAMETERS = enum.auto()
	MAXWELLSIM_INPUTS_CONSTANTS = enum.auto()
	MAXWELLSIM_INPUTS_LISTS = enum.auto()
	
	# Outputs/
	MAXWELLSIM_OUTPUTS = enum.auto()
	MAXWELLSIM_OUTPUTS_VIEWERS = enum.auto()
	MAXWELLSIM_OUTPUTS_EXPORTERS = enum.auto()
	MAXWELLSIM_OUTPUTS_PLOTTERS = enum.auto()
	
	# Sources/
	MAXWELLSIM_SOURCES = enum.auto()
	MAXWELLSIM_SOURCES_TEMPORALSHAPES = enum.auto()
	
	# Mediums/
	MAXWELLSIM_MEDIUMS = enum.auto()
	MAXWELLSIM_MEDIUMS_NONLINEARITIES = enum.auto()
	
	# Structures/
	MAXWELLSIM_STRUCTURES = enum.auto()
	MAXWELLSIM_STRUCTURES_PRIMITIVES = enum.auto()
	
	# Bounds/
	MAXWELLSIM_BOUNDS = enum.auto()
	MAXWELLSIM_BOUNDS_BOUNDFACES = enum.auto()
	
	# Monitors/
	MAXWELLSIM_MONITORS = enum.auto()
	MAXWELLSIM_MONITORS_NEARFIELDPROJECTIONS = enum.auto()
	
	# Simulations/
	MAXWELLSIM_SIMS = enum.auto()
	MAXWELLSIM_SIMGRIDAXES = enum.auto()
	
	# Utilities/
	MAXWELLSIM_UTILITIES = enum.auto()
	MAXWELLSIM_UTILITIES_OPERATIONS = enum.auto()
	
	@classmethod
	def get_tree(cls):
		## TODO: Refactor
		syllable_categories = [
			node_category.value.split("_")
			for node_category in cls
			if node_category.value != "MAXWELLSIM"
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
	NodeCategory.MAXWELLSIM_INPUTS: "Inputs",
	NodeCategory.MAXWELLSIM_INPUTS_SCENE: "Scene",
	NodeCategory.MAXWELLSIM_INPUTS_PARAMETERS: "Parameters",
	NodeCategory.MAXWELLSIM_INPUTS_CONSTANTS: "Constants",
	NodeCategory.MAXWELLSIM_INPUTS_LISTS: "Lists",
	
	# Outputs/
	NodeCategory.MAXWELLSIM_OUTPUTS: "Outputs",
	NodeCategory.MAXWELLSIM_OUTPUTS_VIEWERS: "Viewers",
	NodeCategory.MAXWELLSIM_OUTPUTS_EXPORTERS: "Exporters",
	NodeCategory.MAXWELLSIM_OUTPUTS_PLOTTERS: "Plotters",
	
	# Sources/
	NodeCategory.MAXWELLSIM_SOURCES: "Sources",
	NodeCategory.MAXWELLSIM_SOURCES_TEMPORALSHAPES: "Temporal Shapes",
	
	# Mediums/
	NodeCategory.MAXWELLSIM_MEDIUMS: "Mediums",
	NodeCategory.MAXWELLSIM_MEDIUMS_NONLINEARITIES: "Non-Linearities",
	
	# Structures/
	NodeCategory.MAXWELLSIM_STRUCTURES: "Structures",
	NodeCategory.MAXWELLSIM_STRUCTURES_PRIMITIVES: "Primitives",
	
	# Bounds/
	NodeCategory.MAXWELLSIM_BOUNDS: "Bounds",
	NodeCategory.MAXWELLSIM_BOUNDS_BOUNDFACES: "Bound Faces",
	
	# Monitors/
	NodeCategory.MAXWELLSIM_MONITORS: "Monitors",
	NodeCategory.MAXWELLSIM_MONITORS_NEARFIELDPROJECTIONS: "Near-Field Projections",
	
	# Simulations/
	NodeCategory.MAXWELLSIM_SIMS: "Simulations",
	NodeCategory.MAXWELLSIM_SIMGRIDAXES: "Sim Grid Axes",
	
	# Utilities/
	NodeCategory.MAXWELLSIM_UTILITIES: "Utilities",
	NodeCategory.MAXWELLSIM_UTILITIES_OPERATIONS: "Operations",
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

SocketReturnType = typ.TypeVar('SocketReturnType', covariant=True)
## - Covariance: If B subtypes A, then Container[B] subtypes Container[A].
## - This is absolutely what we want here.

#@typ.runtime_checkable
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
