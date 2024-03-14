from .socket_types import SocketType as ST

BL_SOCKET_DIRECT_TYPE_MAP = {
	("NodeSocketString", 1): ST.String,
	("NodeSocketBool", 1): ST.Bool,
	("NodeSocketCollection", 1): ST.BlenderCollection,
	("NodeSocketImage", 1): ST.BlenderImage,
	("NodeSocketObject", 1): ST.BlenderObject,
	
	("NodeSocketFloat", 1): ST.RealNumber,
	#("NodeSocketFloatAngle", 1): ST.PhysicalAngle,
	#("NodeSocketFloatDistance", 1): ST.PhysicalLength,
	("NodeSocketFloatFactor", 1): ST.RealNumber,
	("NodeSocketFloatPercentage", 1): ST.RealNumber,
	#("NodeSocketFloatTime", 1): ST.PhysicalTime,
	#("NodeSocketFloatTimeAbsolute", 1): ST.PhysicalTime,
	
	("NodeSocketInt", 1): ST.IntegerNumber,
	("NodeSocketIntFactor", 1): ST.IntegerNumber,
	("NodeSocketIntPercentage", 1): ST.IntegerNumber,
	("NodeSocketIntUnsigned", 1): ST.IntegerNumber,
	
	("NodeSocketRotation", 2): ST.PhysicalRot2D,
	("NodeSocketColor", 3): ST.Color,
	("NodeSocketVector", 2): ST.Real2DVector,
	("NodeSocketVector", 3): ST.Real3DVector,
	#("NodeSocketVectorAcceleration", 2): ST.PhysicalAccel2D,
	#("NodeSocketVectorAcceleration", 3): ST.PhysicalAccel3D,
	#("NodeSocketVectorDirection", 2): ST.Real2DVectorDir,
	#("NodeSocketVectorDirection", 3): ST.Real3DVectorDir,
	("NodeSocketVectorEuler", 2): ST.PhysicalRot2D,
	("NodeSocketVectorEuler", 3): ST.PhysicalRot3D,
	#("NodeSocketVectorTranslation", 3): ST.PhysicalDisp3D,
	#("NodeSocketVectorVelocity", 3): ST.PhysicalVel3D,
	#("NodeSocketVectorXYZ", 3): ST.PhysicalPoint3D,
}