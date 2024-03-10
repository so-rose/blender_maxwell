import sympy.physics.units as spu
from ....utils import extra_sympy_units as spuex

from .socket_types import SocketType as ST

Dimensions = int  ## Num. Elements in the Socket
BLSocketType = str  ## A Blender-Defined Socket Type

class BLSocketToSocket:
	"""Encodes ways of converting blender sockets of known dimensionality
	to a corresponding SocketType.
	
	"Dimensionality" is simply how many elements the blender socket has.
	The user must explicitly specify this, as blender allows a variable
	number of elements for some sockets, and we do not.
	"""
	####################
	# - Direct BLSocketType -> SocketType
	####################
	by_bl_socket_type: dict[Dimensions, dict[BLSocketType, ST]] = {
		1: {
			"NodeSocketStandard": ST.Any,
			"NodeSocketVirtual": ST.Any,
			"NodeSocketGeometry": ST.Any,
			"NodeSocketTexture": ST.Any,
			"NodeSocketShader": ST.Any,
			"NodeSocketMaterial": ST.Any,
			
			"NodeSocketString": ST.Text,
			"NodeSocketBool": ST.Bool,
			"NodeSocketCollection": ST.BlenderCollection,
			"NodeSocketImage": ST.BlenderImage,
			"NodeSocketObject": ST.BlenderObject,
			
			"NodeSocketFloat": ST.RealNumber,
			"NodeSocketFloatAngle": ST.PhysicalAngle,
			"NodeSocketFloatDistance": ST.PhysicalLength,
			"NodeSocketFloatFactor": ST.RealNumber,
			"NodeSocketFloatPercentage": ST.RealNumber,
			"NodeSocketFloatTime": ST.PhysicalTime,
			"NodeSocketFloatTimeAbsolute": ST.RealNumber,
			"NodeSocketFloatUnsigned": ST.RealNumber,
			
			"NodeSocketInt": ST.IntegerNumber,
			"NodeSocketIntFactor": ST.IntegerNumber,
			"NodeSocketIntPercentage": ST.IntegerNumber,
			"NodeSocketIntUnsigned": ST.IntegerNumber,
		},
		2: {
			"NodeSocketVector": ST.Real3DVector,
			"NodeSocketVectorAcceleration": ST.Real3DVector,
			"NodeSocketVectorDirection": ST.Real3DVector,
			"NodeSocketVectorEuler": ST.Real3DVector,
			"NodeSocketVectorTranslation": ST.Real3DVector,
			"NodeSocketVectorVelocity": ST.Real3DVector,
			"NodeSocketVectorXYZ": ST.Real3DVector,
			#"NodeSocketVector": ST.Real2DVector,
			#"NodeSocketVectorAcceleration": ST.PhysicalAccel2D,
			#"NodeSocketVectorDirection": ST.PhysicalDir2D,
			#"NodeSocketVectorEuler": ST.PhysicalEuler2D,
			#"NodeSocketVectorTranslation": ST.PhysicalDispl2D,
			#"NodeSocketVectorVelocity": ST.PhysicalVel2D,
			#"NodeSocketVectorXYZ": ST.Real2DPoint,
		},
		3: {
			"NodeSocketRotation": ST.Real3DVector,
			
			"NodeSocketColor": ST.Any,
			
			"NodeSocketVector": ST.Real3DVector,
			#"NodeSocketVectorAcceleration": ST.PhysicalAccel3D,
			#"NodeSocketVectorDirection": ST.PhysicalDir3D,
			#"NodeSocketVectorEuler": ST.PhysicalEuler3D,
			#"NodeSocketVectorTranslation": ST.PhysicalDispl3D,
			"NodeSocketVectorTranslation": ST.PhysicalPoint3D,
			#"NodeSocketVectorVelocity": ST.PhysicalVel3D,
			"NodeSocketVectorXYZ": ST.PhysicalPoint3D,
		},
	}
	
	####################
	# - BLSocket Description-Driven SocketType Choice
	####################
	by_description = {
		1: {
			"Angle": ST.PhysicalAngle,
			
			"Length": ST.PhysicalLength,
			"Area": ST.PhysicalArea,
			"Volume": ST.PhysicalVolume,
			
			"Mass": ST.PhysicalMass,
			
			"Speed": ST.PhysicalSpeed,
			"Accel": ST.PhysicalAccelScalar,
			"Force": ST.PhysicalForceScalar,
			
			"Freq": ST.PhysicalFreq,
		},
		2: {
			#"2DCount": ST.Int2DVector,
			
			#"2DPoint": ST.PhysicalPoint2D,
			#"2DSize": ST.PhysicalSize2D,
			#"2DPol": ST.PhysicalPol,
			"2DPoint": ST.PhysicalPoint3D,
			"2DSize": ST.PhysicalSize3D,
		},
		3: {
			#"Count": ST.Int3DVector,
			
			"Point": ST.PhysicalPoint3D,
			"Size": ST.PhysicalSize3D,
			
			#"Force": ST.PhysicalForce3D,
			
			"Freq": ST.PhysicalSize3D,
		},
	}
