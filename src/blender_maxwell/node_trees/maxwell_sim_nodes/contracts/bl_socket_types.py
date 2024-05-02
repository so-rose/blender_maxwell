import dataclasses
import enum
import typing as typ

import bpy
import sympy as sp

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from .socket_types import SocketType

log = logger.get(__name__)

BL_SOCKET_DESCR_ANNOT_STRING = ':: '


@dataclasses.dataclass(kw_only=True, frozen=True)
class BLSocketInfo:
	has_support: bool
	is_preview: bool
	socket_type: SocketType | None
	size: spux.NumberSize1D | None
	physical_type: spux.PhysicalType | None
	default_value: spux.ScalarUnitlessRealExpr

	bl_isocket_identifier: spux.ScalarUnitlessRealExpr


class BLSocketType(enum.StrEnum):
	Virtual = 'NodeSocketVirtual'
	# Blender
	Image = 'NodeSocketImage'
	Shader = 'NodeSocketShader'
	Material = 'NodeSocketMaterial'
	Geometry = 'NodeSocketGeometry'
	Object = 'NodeSocketObject'
	Collection = 'NodeSocketCollection'
	# Basic
	Bool = 'NodeSocketBool'
	String = 'NodeSocketString'
	Menu = 'NodeSocketMenu'
	# Float
	Float = 'NodeSocketFloat'
	FloatUnsigned = 'NodeSocketFloatUnsigned'
	FloatAngle = 'NodeSocketFloatAngle'
	FloatDistance = 'NodeSocketFloatDistance'
	FloatFactor = 'NodeSocketFloatFactor'
	FloatPercentage = 'NodeSocketFloatPercentage'
	FloatTime = 'NodeSocketFloatTime'
	FloatTimeAbsolute = 'NodeSocketFloatTimeAbsolute'
	# Int
	Int = 'NodeSocketInt'
	IntFactor = 'NodeSocketIntFactor'
	IntPercentage = 'NodeSocketIntPercentage'
	IntUnsigned = 'NodeSocketIntUnsigned'
	# Vector
	Color = 'NodeSocketColor'
	Rotation = 'NodeSocketRotation'
	Vector = 'NodeSocketVector'
	VectorAcceleration = 'NodeSocketAcceleration'
	VectorDirection = 'NodeSocketDirection'
	VectorEuler = 'NodeSocketEuler'
	VectorTranslation = 'NodeSocketTranslation'
	VectorVelocity = 'NodeSocketVelocity'
	VectorXYZ = 'NodeSocketXYZ'

	@staticmethod
	def from_bl_isocket(
		bl_isocket: bpy.types.NodeTreeInterfaceSocket,
	) -> typ.Self:
		return BLSocketType(bl_isocket.bl_socket_idname)

	@staticmethod
	def info_from_bl_isocket(
		bl_isocket: bpy.types.NodeTreeInterfaceSocket,
	) -> typ.Self:
		bl_socket_type = BLSocketType.from_bl_isocket(bl_isocket)
		if bl_socket_type.has_support:
			return bl_socket_type.parse(
				bl_isocket.default_value, bl_isocket.description, bl_isocket.identifier
			)
		return bl_socket_type.parse(None, bl_isocket.description, bl_isocket.identifier)

	####################
	# - Direct Properties
	####################
	@property
	def has_support(self) -> bool:
		BLST = BLSocketType
		return {
			BLST.Virtual: False,
			BLST.Geometry: False,
			BLST.Shader: False,
			BLST.FloatUnsigned: False,
			BLST.IntUnsigned: False,
		}.get(self, True)

	@property
	def socket_type(self) -> SocketType | None:
		"""Deduce `SocketType` corresponding to the Blender socket type.

		**The socket type alone is not enough** to actually create the socket.
		To declare a socket in the addon, an appropriate `socket.SocketDef` must be constructed in a manner that respects contextual information.

		Returns:
			The corresponding socket type, if the addon has support for mapping this Blender socket.
			For sockets with support, the fallback is always `SocketType.Expr`.

			Support is determined using `self.has_support`
		"""
		if not self.has_support:
			return None

		BLST = BLSocketType
		ST = SocketType
		return {
			# Blender
			# Basic
			BLST.Bool: ST.String,
			# Float
			# Array-Like
			BLST.Color: ST.Color,
		}.get(self, ST.Expr)

	@property
	def mathtype(self) -> spux.MathType | None:
		"""Deduce `spux.MathType` corresponding to the Blender socket type.

		**The socket type alone is not enough** to actually create the socket.
		To declare a socket in the addon, an appropriate `socket.SocketDef` must be constructed in a manner that respects contextual information.

		Returns:
			The corresponding socket type, if the addon has support for mapping this Blender socket.
			For sockets with support, the fallback is always `SocketType.Expr`.

			Support is determined using `self.has_support`
		"""
		if not self.has_support:
			return None

		BLST = BLSocketType
		MT = spux.MathType
		return {
			# Blender
			# Basic
			BLST.Bool: MT.Bool,
			# Float
			BLST.Float: MT.Real,
			BLST.FloatAngle: MT.Real,
			BLST.FloatDistance: MT.Real,
			BLST.FloatFactor: MT.Real,
			BLST.FloatPercentage: MT.Real,
			BLST.FloatTime: MT.Real,
			BLST.FloatTimeAbsolute: MT.Real,
			# Int
			BLST.Int: MT.Integer,
			BLST.IntFactor: MT.Integer,
			BLST.IntPercentage: MT.Integer,
			# Vector
			BLST.Color: MT.Real,
			BLST.Rotation: MT.Real,
			BLST.Vector: MT.Real,
			BLST.VectorAcceleration: MT.Real,
			BLST.VectorDirection: MT.Real,
			BLST.VectorEuler: MT.Real,
			BLST.VectorTranslation: MT.Real,
			BLST.VectorVelocity: MT.Real,
			BLST.VectorXYZ: MT.Real,
		}.get(self)

	@property
	def size(
		self,
	) -> (
		typ.Literal[
			spux.NumberSize1D.Scalar, spux.NumberSize1D.Vec3, spux.NumberSize1D.Vec4
		]
		| None
	):
		"""Deduce the internal size of the Blender socket's data.

		Returns:
			A `spux.NumberSize1D` reflecting the internal data representation.
			Always falls back to `{spux.NumberSize1D.Scalar}`.
		"""
		if not self.has_support:
			return None

		S = spux.NumberSize1D
		BLST = BLSocketType
		return {
			BLST.Color: S.Vec4,
			BLST.Rotation: S.Vec3,
			BLST.Vector: S.Vec3,
			BLST.VectorAcceleration: S.Vec3,
			BLST.VectorDirection: S.Vec3,
			BLST.VectorEuler: S.Vec3,
			BLST.VectorTranslation: S.Vec3,
			BLST.VectorVelocity: S.Vec3,
			BLST.VectorXYZ: S.Vec3,
		}.get(self, S.Scalar)

	@property
	def unambiguous_physical_type(self) -> spux.PhysicalType | None:
		"""Deduce an **unambiguous** physical type from the Blender socket, if any.

		Blender does have its own unit systems, which leads to some Blender socket subtypes having an obvious choice of physical unit dimension (ex. `BLSocketType.FloatTime`).
		In such cases, the `spux.PhysicalType` that matches the Blender socket can be uniquely determined.

		When a phsyical type cannot be immediately determined in this way, other mechanisms must be used to deduce what to do.

		Returns:
			A physical type corresponding to the Blender socket, **if exactly one** can be determined with no ambiguity - else `None`.

			If more than one physical type might apply,  `None`.
		"""
		if not self.has_support:
			return None

		P = spux.PhysicalType
		BLST = BLSocketType
		{
			BLST.FloatAngle: P.Angle,
			BLST.FloatDistance: P.Length,
			BLST.FloatTime: P.Time,
			BLST.FloatTimeAbsolute: P.Time,  ## What's the difference?
			BLST.VectorAcceleration: P.Accel,
			## BLST.VectorDirection: Directions are unitless (within cartesian)
			BLST.VectorEuler: P.Angle,
			BLST.VectorTranslation: P.Length,
			BLST.VectorVelocity: P.Vel,
			BLST.VectorXYZ: P.Length,
		}.get(self)

	@property
	def valid_sizes(self) -> set[spux.NumberSize1D] | None:
		"""Deduce which sizes it would be valid to interpret a Blender socket as having.

		This property's purpose is merely to present a set of options that _are valid_.
		Whether an a size _is truly usable_, can only be determined with contextual information, wherein certain decisions can be made:

		- **2D vs. 3D**:  In general, Blender's vector socket types are **only** 3D, and we cannot ever _really_ have a 2D vector.
			If one chooses interpret ex. `BLSocketType.Vector` as 2D, one might do so by pretending the third coordinate doesn't exist.
			But **this is a subjective decision**, which always has to align with the logic on the other side of the Blender socket.
		- **Colors**: Generally, `BLSocketType.Color` is always 4D, representing `RGBA` (with alpha channel).
			However, we often don't care about alpha; therefore, we might choose to "just" push a 3D `RGB` vector.
			Again, **this is a subjective decision** which requires one to make a decision about alpha, for example "alpha is always 1".
		- **Scalars**: We can generally always interpret a scalar as a vector, using well-defined "broadcasting".

		Returns:
			The set of `spux.NumberSize1D`s, which it would be safe to interpret the Blender socket as having.

			Always falls back to `{spux.NumberSize1D.Scalar}`.
		"""
		if not self.has_support:
			return None

		S = spux.NumberSize1D
		BLST = BLSocketType
		return {
			BLST.Color: {S.Scalar, S.Vec3, S.Vec4},
			BLST.Rotation: {S.Vec2, S.Vec3},
			BLST.VectorAcceleration: {S.Scalar, S.Vec2, S.Vec3},
			BLST.VectorDirection: {S.Scalar, S.Vec2, S.Vec3},
			BLST.VectorEuler: {S.Vec2, S.Vec3},
			BLST.VectorTranslation: {S.Scalar, S.Vec2, S.Vec3},
			BLST.VectorVelocity: {S.Scalar, S.Vec2, S.Vec3},
			BLST.VectorXYZ: {S.Scalar, S.Vec2, S.Vec3},
		}.get(self, {S.Scalar})

	####################
	# - Parsed Properties
	####################
	def parse(
		self, bl_default_value: typ.Any, description: str, bl_isocket_identifier: str
	) -> BLSocketInfo:
		# Parse the Description
		## TODO: Some kind of error on invalid parse if there is also no unambiguous physical type
		descr_params = description.split(BL_SOCKET_DESCR_ANNOT_STRING)[0]
		directive = (
			_tokens[0]
			if (_tokens := descr_params.split(' '))[0] != '2D'
			else _tokens[1]
		)

		## Interpret the Description Parse
		parsed_physical_type = getattr(spux.PhysicalType, directive, None)
		physical_type = (
			self.unambiguous_physical_type
			if self.unambiguous_physical_type is not None
			else parsed_physical_type
		)

		# Parse the Default Value
		if self.mathtype is not None and bl_default_value is not None:
			if self.size == spux.NumberSize1D.Scalar:
				default_value = self.mathtype.pytype(bl_default_value)
			elif description.startswith('2D'):
				default_value = sp.Matrix(tuple(bl_default_value)[:2])
			else:
				default_value = sp.Matrix(tuple(bl_default_value))
		else:
			default_value = bl_default_value

		# Return Parsed Socket Information
		## -> Combining directly known and parsed knowledge.
		## -> Should contain everything needed to match the Blender socket.
		return BLSocketInfo(
			has_support=self.has_support,
			is_preview=(directive == 'Preview'),
			socket_type=self.socket_type,
			size=self.size,
			physical_type=physical_type,
			default_value=default_value,
			bl_isocket_identifier=bl_isocket_identifier,
		)
