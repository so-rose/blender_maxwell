# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
	mathtype: spux.MathType | None
	physical_type: spux.PhysicalType | None
	default_value: spux.ScalarUnitlessRealExpr

	bl_isocket_identifier: spux.ScalarUnitlessRealExpr

	def encode(
		self, raw_value: typ.Any, unit_system: spux.UnitSystem | None
	) -> typ.Any:
		"""Encode a raw value, given a unit system, to be directly writable to a node socket.

		This encoded form is also guaranteed to support writing to a node socket via a modifier interface.
		"""
		# Non-Numerical: Passthrough
		if unit_system is None or self.physical_type is None:
			return raw_value

		# Numerical: Convert to Pure Python Type
		if (
			unit_system is not None
			and self.physical_type is not spux.PhysicalType.NonPhysical
		):
			unitless_value = spux.scale_to_unit_system(raw_value, unit_system)
		elif isinstance(raw_value, spux.SympyType):
			unitless_value = spux.sympy_to_python(raw_value)
		else:
			unitless_value = raw_value

		# Coerce int -> float w/Target is Real
		## -> The value - modifier - GN path is more strict than properties.
		if self.mathtype is spux.MathType.Real and isinstance(unitless_value, int):
			return float(unitless_value)

		return unitless_value


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
		"""Deduce the exact `BLSocketType` represented by an interface socket.

		Interface sockets are an abstraction of what any instance of a particular node tree _will have of input sockets_ once constructed.
		"""
		return BLSocketType(bl_isocket.bl_socket_idname)

	@staticmethod
	def info_from_bl_isocket(
		bl_isocket: bpy.types.NodeTreeInterfaceSocket,
	) -> typ.Self:
		"""Deduce all `BLSocketInfo` from an interface socket.

		This is a high-level method providing a clean way to chain `BLSocketType.from_bl_isocket()` together with `self.parse()`.
		"""
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
		"""Decides whether the current `BLSocketType` is explicitly not supported.

		Not all socket types make sense to represent in our node tree.
		In general, these should be skipped.
		"""
		BLST = BLSocketType
		return {
			# Won't Fix
			BLST.Virtual: False,
			BLST.Geometry: False,
			BLST.Shader: False,
			BLST.FloatUnsigned: False,
			BLST.IntUnsigned: False,
			## TODO
			BLST.Menu: False,
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
			BLST.Image: ST.BlenderImage,
			BLST.Material: ST.BlenderMaterial,
			BLST.Object: ST.BlenderObject,
			BLST.Collection: ST.BlenderCollection,
			# Basic
			BLST.Bool: ST.BlenderCollection,
			BLST.String: ST.String,
			# Float
			BLST.Float: ST.Expr,
			BLST.FloatAngle: ST.Expr,
			BLST.FloatDistance: ST.Expr,
			BLST.FloatFactor: ST.Expr,
			BLST.FloatPercentage: ST.Expr,
			BLST.FloatTime: ST.Expr,
			BLST.FloatTimeAbsolute: ST.Expr,
			# Int
			BLST.Int: ST.Expr,
			BLST.IntFactor: ST.Expr,
			BLST.IntPercentage: ST.Expr,
			# Vector
			BLST.Color: ST.Color,
			BLST.Rotation: ST.Expr,
			BLST.Vector: ST.Expr,
			BLST.VectorAcceleration: ST.Expr,
			BLST.VectorDirection: ST.Expr,
			BLST.VectorEuler: ST.Expr,
			BLST.VectorTranslation: ST.Expr,
			BLST.VectorVelocity: ST.Expr,
			BLST.VectorXYZ: ST.Expr,
		}[self]

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
			BLST.FloatTimeAbsolute: P.Time,  ## TODO: What's the difference?
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
		# Unpack Description
		## -> TODO: Raise an kind of error on invalid parse if there is also no unambiguous physical type
		descr_params = description.split(BL_SOCKET_DESCR_ANNOT_STRING)[0]
		directive = (
			_tokens[0]
			if (_tokens := descr_params.split(' '))[0] != '2D'
			else _tokens[1]
		)

		# Parse PhysicalType
		## -> None if there is no appropriate MathType.
		## -> Otherwise, prefer unambiguous - description hint - NonPhysical
		has_physical_type = self.mathtype in [
			spux.MathType.Integer,
			spux.MathType.Rational,
			spux.MathType.Real,
			spux.MathType.Complex,
		]
		if has_physical_type:
			parsed_physical_type = getattr(spux.PhysicalType, directive, None)
			physical_type = (
				self.unambiguous_physical_type
				if self.unambiguous_physical_type is not None
				else (
					parsed_physical_type
					if parsed_physical_type is not None
					else spux.PhysicalType.NonPhysical
				)
			)
		else:
			physical_type = None

		# Parse Default Value
		## -> Read the Blender socket's default value and convrt it
		if self.mathtype is not None and bl_default_value is not None:
			# Scalar: Convert to Pure Python TYpe
			if self.size == spux.NumberSize1D.Scalar:
				default_value = self.mathtype.pytype(bl_default_value)

			# 2D (Description Hint): Sympy Matrix
			## -> The description hint "2D" is the trigger for this.
			## -> Ignore the last component to get the effect of "2D".
			elif description.startswith('2D'):
				default_value = sp.ImmutableMatrix(tuple(bl_default_value)[:2])

			# 3D/4D: Simple Parse to Sympy Matrix
			## -> We don't explicitly check the size.
			else:
				default_value = sp.ImmutableMatrix(tuple(bl_default_value))

		else:
			# Non-Mathematical: Passthrough
			default_value = bl_default_value

		# Return Parsed Socket Information
		## -> Combining directly known and parsed knowledge.
		## -> Should contain everything needed to create a socket in our tree.
		return BLSocketInfo(
			has_support=self.has_support,
			is_preview=(directive == 'Preview'),
			socket_type=self.socket_type,
			size=self.size,
			mathtype=self.mathtype,
			physical_type=physical_type,
			default_value=default_value,
			bl_isocket_identifier=bl_isocket_identifier,
		)
