import sympy.physics.units as spu
from ....utils import extra_sympy_units as spuex

from .socket_types import SocketType as ST

## TODO: Don't just presume sRGB.
SOCKET_COLORS = {
	# Basic
	ST.Any: (0.8, 0.8, 0.8, 1.0),  # Light Grey
	ST.Bool: (0.7, 0.7, 0.7, 1.0),  # Medium Light Grey
	ST.Text: (0.7, 0.7, 0.7, 1.0),  # Medium Light Grey
	ST.FilePath: (0.6, 0.6, 0.6, 1.0),  # Medium Grey
	ST.Secret: (0.0, 0.0, 0.0, 1.0),  # Black

	# Number
	ST.IntegerNumber: (0.5, 0.5, 1.0, 1.0),  # Light Blue
	ST.RationalNumber: (0.4, 0.4, 0.9, 1.0),  # Medium Light Blue
	ST.RealNumber: (0.3, 0.3, 0.8, 1.0),  # Medium Blue
	ST.ComplexNumber: (0.2, 0.2, 0.7, 1.0),  # Dark Blue

	# Vector
	ST.Real2DVector: (0.5, 1.0, 0.5, 1.0),  # Light Green
	ST.Complex2DVector: (0.4, 0.9, 0.4, 1.0),  # Medium Light Green
	ST.Real3DVector: (0.3, 0.8, 0.3, 1.0),  # Medium Green
	ST.Complex3DVector: (0.2, 0.7, 0.2, 1.0),  # Dark Green
	
	# Physical
	ST.PhysicalUnitSystem: (1.0, 0.5, 0.5, 1.0),  # Light Red
	ST.PhysicalTime: (1.0, 0.5, 0.5, 1.0),  # Light Red
	ST.PhysicalAngle: (0.9, 0.45, 0.45, 1.0),  # Medium Light Red
	ST.PhysicalLength: (0.8, 0.4, 0.4, 1.0),  # Medium Red
	ST.PhysicalArea: (0.7, 0.35, 0.35, 1.0),  # Medium Dark Red
	ST.PhysicalVolume: (0.6, 0.3, 0.3, 1.0),  # Dark Red
	ST.PhysicalPoint2D: (0.7, 0.35, 0.35, 1.0),  # Medium Dark Red
	ST.PhysicalPoint3D: (0.6, 0.3, 0.3, 1.0),  # Dark Red
	ST.PhysicalSize2D: (0.7, 0.35, 0.35, 1.0),  # Medium Dark Red
	ST.PhysicalSize3D: (0.6, 0.3, 0.3, 1.0),  # Dark Red
	ST.PhysicalMass: (0.9, 0.6, 0.4, 1.0),  # Light Orange
	ST.PhysicalSpeed: (0.8, 0.55, 0.35, 1.0),  # Medium Light Orange
	ST.PhysicalAccelScalar: (0.7, 0.5, 0.3, 1.0),  # Medium Orange
	ST.PhysicalForceScalar: (0.6, 0.45, 0.25, 1.0),  # Medium Dark Orange
	ST.PhysicalAccel3DVector: (0.7, 0.5, 0.3, 1.0),  # Medium Orange
	ST.PhysicalForce3DVector: (0.6, 0.45, 0.25, 1.0),  # Medium Dark Orange
	ST.PhysicalPol: (0.5, 0.4, 0.2, 1.0),  # Dark Orange
	ST.PhysicalFreq: (1.0, 0.7, 0.5, 1.0),  # Light Peach

	# Blender
	ST.BlenderObject: (0.7, 0.5, 1.0, 1.0),  # Light Purple
	ST.BlenderCollection: (0.6, 0.45, 0.9, 1.0),  # Medium Light Purple
	ST.BlenderImage: (0.5, 0.4, 0.8, 1.0),  # Medium Purple
	ST.BlenderVolume: (0.4, 0.35, 0.7, 1.0),  # Medium Dark Purple
	ST.BlenderGeoNodes: (0.3, 0.3, 0.6, 1.0),  # Dark Purple
	ST.BlenderText: (0.5, 0.5, 0.75, 1.0),  # Light Lavender
	ST.BlenderPreviewTarget: (0.5, 0.5, 0.75, 1.0),  # Light Lavender

	# Maxwell
	ST.MaxwellSource: (1.0, 1.0, 0.5, 1.0),  # Light Yellow
	ST.MaxwellTemporalShape: (0.9, 0.9, 0.45, 1.0),  # Medium Light Yellow
	ST.MaxwellMedium: (0.8, 0.8, 0.4, 1.0),  # Medium Yellow
	ST.MaxwellMediumNonLinearity: (0.7, 0.7, 0.35, 1.0),  # Medium Dark Yellow
	ST.MaxwellStructure: (0.6, 0.6, 0.3, 1.0),  # Dark Yellow
	ST.MaxwellBoundBox: (0.9, 0.8, 0.5, 1.0),  # Light Gold
	ST.MaxwellBoundFace: (0.8, 0.7, 0.45, 1.0),  # Medium Light Gold
	ST.MaxwellMonitor: (0.7, 0.6, 0.4, 1.0),  # Medium Gold
	ST.MaxwellFDTDSim: (0.6, 0.5, 0.35, 1.0),  # Medium Dark Gold
	ST.MaxwellSimGrid: (0.5, 0.4, 0.3, 1.0),  # Dark Gold
	ST.MaxwellSimGridAxis: (0.4, 0.3, 0.25, 1.0),  # Darkest Gold
	ST.MaxwellSimDomain: (0.4, 0.3, 0.25, 1.0),  # Darkest Gold
	
	# Tidy3D
	ST.Tidy3DCloudTask: (0.4, 0.3, 0.25, 1.0),  # Darkest Gold
}

