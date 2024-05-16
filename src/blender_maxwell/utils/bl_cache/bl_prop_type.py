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

"""Defines `BLPropType`, which provides stronger lower-level interfaces for interacting with data that can be conformed to work with Blender properties."""

import builtins
import enum
import functools
import inspect
import pathlib
import typing as typ
from pathlib import Path

import bpy
import numpy as np

from blender_maxwell import contracts as ct
from blender_maxwell.utils import logger, serialize
from blender_maxwell.utils.staticproperty import staticproperty

from .signal import Signal

log = logger.get(__name__)

####################
# - Types
####################
BLIDStructs = typ.get_args(ct.BLIDStruct)
Shape: typ.TypeAlias = None | tuple[int, ...]
BLPropInfo: typ.TypeAlias = dict[str, typ.Any]


@functools.cache
def _parse_vector_size(obj_type: type[tuple[int, ...]]) -> int:
	"""Parse the size of an arbitrarily sized generic tuple type, which is representing a vector.

	Parameters:
		obj_type: The type of a flat, generic tuple integer, representing a static vector shape.

	Returns:
		The length of any object that has the given type.
	"""
	return len(typ.get_args(obj_type))


@functools.cache
def _parse_matrix_size(obj_type: type[tuple[int, ...], ...]) -> tuple[int, int]:
	"""Parse the rows and columns of an arbitrarily sized generic tuple-of-tuple type, which is representing a row-major matrix.

	Parameters:
		obj_type: The type of a singly-nested, generic tuple integer, representing a static matrix shape.

	Returns:
		The rows and columns of any object that has the given type.
	"""
	rows = len(typ.get_args(obj_type))
	cols = len(typ.get_args(typ.get_args(obj_type)[0]))

	for i, col_generic in enumerate(typ.get_args(obj_type)):
		col_els = typ.get_args(col_generic)
		if len(col_els) != cols:
			msg = f'Value {obj_type} has mismatching column length {i} (to column 0)'
			raise ValueError(msg)

	return (rows, cols)


def _is_strenum(T: type) -> bool:  # noqa: N803
	return inspect.isclass(T) and issubclass(T, enum.StrEnum)


####################
# - Blender Property Type
####################
class BLPropType(enum.StrEnum):
	"""A type identifier which can be directly associated with a Blender property.

	For general use, the higher-level interface `BLProp` is more appropriate.

	This is a low-level interface to Blender properties, allowing for directly identifying and utilizing a subset of types that are trivially representable using Blender's property system.
	This hard-coded approach is especially required when managing the nuances of UI methods.

	`BLPropType` should generally be treated as a "dumb" enum identifying the low-level representation of an object in a Blender property.
	Use of `BLPropType.from_type` is encouraged; use of other methods is generally discouraged outside of higher-level encapsulating interfaces.


	Attributes:
		Bool: A boolean.
		Int: An integer.
		Float: A floating point number.
		BoolVector: Between 2 and 32 booleans.
		IntVector: Between 2 and 32 integers.
		FloatVector: Between 2 and 32 floats.
		BoolVector: 2D booleans of 2 - 32 elements per axis.
		IntVector: 2D integers of 2 - 32 elements per axis.
		FloatVector: 2D floats of 2 - 32 elements per axis.
		Str: A simple string.
		Path: A particular filesystem path.
		SingleEnum: A single string value from a statically known `StrEnum`.
		SetEnum: A set of string values, each from a statically known `StrEnum`.
		SingleDynEnum: A single string value from a dynamically computed set of string values.
		SetDynEnum: A set of string value, each from a dynamically computed set of string values.
		BLPointer: A reference to a Blender object.
			Blender manages correctly reconstructing this reference on startup, and the underlying pointer value is not stable.
		Serialized: An arbitrary, serialized representation of an object.
	"""

	# Scalar
	Bool = enum.auto()
	Int = enum.auto()
	Float = enum.auto()
	## TODO: Support complex

	# Vector
	BoolVector = enum.auto()
	IntVector = enum.auto()
	FloatVector = enum.auto()

	# Matrix
	BoolMatrix = enum.auto()
	IntMatrix = enum.auto()
	FloatMatrix = enum.auto()

	## TODO: Support jaxtyping JAX arrays (as serialized) directly?

	# String
	Str = enum.auto()
	Path = enum.auto()
	## TODO: OS checks for Path

	# Enums
	SingleEnum = enum.auto()
	SetEnum = enum.auto()

	SingleDynEnum = enum.auto()
	SetDynEnum = enum.auto()

	# Special
	BLPointer = enum.auto()
	Serialized = enum.auto()

	####################
	# - Static
	####################
	@staticproperty
	def vector_types() -> frozenset[typ.Self]:
		"""The set of `BLPropType`s that are considered "vectors"."""
		BPT = BLPropType
		return frozenset({BPT.BoolVector, BPT.IntVector, BPT.FloatVector})

	@staticproperty
	def matrix_types() -> frozenset[typ.Self]:
		"""The set of `BLPropType`s that are considered "matrices"."""
		BPT = BLPropType
		return frozenset({BPT.BoolMatrix, BPT.IntMatrix, BPT.FloatMatrix})

	####################
	# - Computed
	####################
	@functools.cached_property
	def is_vector(self) -> bool:
		"""Checks whether this `BLPropType` is considered a vector.

		Returns:
			A boolean indicating "vectorness".
		"""
		return self in BLPropType.vector_types

	@functools.cached_property
	def is_matrix(self) -> bool:
		"""Checks whether this `BLPropType` is considered a matrix.

		Returns:
			A boolean indicating "matrixness".
		"""
		return self in BLPropType.matrix_types

	@functools.cached_property
	def bl_prop(self) -> bpy.types.Property:
		"""Deduce which `bpy.props.*` type should implement this `BLPropType` in practice.

		In practice, `self.parse_kwargs()` collects arguments usable by the type returned by this property.
		Thus, this property provides the key bridge between `BLPropType` and vanilla Blender properties.

		Returns:
			A Blender property type, for use as a constructor.
		"""
		BPT = BLPropType
		return {
			# Scalar
			BPT.Bool: bpy.props.BoolProperty,
			BPT.Int: bpy.props.IntProperty,
			BPT.Float: bpy.props.FloatProperty,
			# Vector
			BPT.BoolVector: bpy.props.BoolVectorProperty,
			BPT.IntVector: bpy.props.IntVectorProperty,
			BPT.FloatVector: bpy.props.FloatVectorProperty,
			# Matrix
			BPT.BoolMatrix: bpy.props.BoolVectorProperty,
			BPT.IntMatrix: bpy.props.IntVectorProperty,
			BPT.FloatMatrix: bpy.props.FloatVectorProperty,
			# String
			BPT.Str: bpy.props.StringProperty,
			BPT.Path: bpy.props.StringProperty,
			# Enum
			BPT.SingleEnum: bpy.props.EnumProperty,
			BPT.SetEnum: bpy.props.EnumProperty,
			BPT.SingleDynEnum: bpy.props.EnumProperty,
			BPT.SetDynEnum: bpy.props.EnumProperty,
			# Special
			BPT.BLPointer: bpy.props.PointerProperty,
			BPT.Serialized: bpy.props.StringProperty,
		}[self]

	@functools.cached_property
	def primitive_type(self) -> type:
		"""The "primitive" type representable using this property.

		Generally, "primitive" types are Python standard library types.
		However, exceptions may exist for a ubiquitously used type.

		Returns:
			A type guaranteed to be representable as a Blender property via. `self.encode()`.

			Note that any relevant constraints on the type are not taken into account in this type.
			For example, `SingleEnum` has `str`, even though all strings are not valid.
			Similarly for ex. non-negative integers simply returning `int`.
		"""
		BPT = BLPropType
		return {
			# Scalar
			BPT.Bool: bool,
			BPT.Int: int,
			BPT.Float: float,
			# Vector
			BPT.BoolVector: bool,
			BPT.IntVector: int,
			BPT.FloatVector: float,
			# Matrix
			BPT.BoolMatrix: bool,
			BPT.IntMatrix: int,
			BPT.FloatMatrix: float,
			# String
			BPT.Str: str,
			BPT.Path: Path,
			# Enum
			BPT.SingleEnum: str,
			BPT.SetEnum: set[str],
			BPT.SingleDynEnum: str,
			BPT.SetDynEnum: set[str],
			# Special
			BPT.BLPointer: None,
			BPT.Serialized: str,
		}[self]

	####################
	# - Parser Methods
	####################
	def parse_size(self, obj_type: type) -> Shape:
		"""Retrieve the shape / shape of data associated with this `BLPropType`.

		Returns:
			Vectors have `(size,)`.
			Matrices have `(rows, cols)`.

			Otherwise, `None` indicates a single value/scalar.
		"""
		BPT = BLPropType

		match self:
			case BPT.BoolVector | BPT.IntVector | BPT.FloatVector:
				return _parse_vector_size(obj_type)
			case BPT.BoolMatrix | BPT.IntMatrix | BPT.FloatMatrix:
				return _parse_matrix_size(obj_type)
			case _:
				return None

	####################
	# - KWArg Parsers
	####################
	@functools.cached_property
	def required_info(self) -> list[str]:
		"""Retrieve a list of required keyword arguments to the constructor returned by `self.bl_prop`.

		Mainly useful via `self.check_info()`.

		Returns:
			A list of required keys for the Blender property constructor.
		"""
		BPT = BLPropType
		return {
			# Scalar
			BPT.Bool: ['default'],
			BPT.Int: ['default'],
			BPT.Float: ['default'],
			# Vector
			BPT.BoolVector: ['default'],
			BPT.IntVector: ['default'],
			BPT.FloatVector: ['default'],
			# Matrix
			BPT.BoolMatrix: ['default'],
			BPT.IntMatrix: ['default'],
			BPT.FloatMatrix: ['default'],
			# String
			BPT.Str: ['default', 'str_search'],
			BPT.Path: ['default', 'path_type'],
			# Enum
			BPT.SingleEnum: ['default'],
			BPT.SetEnum: ['default'],
			BPT.SingleDynEnum: ['enum_dynamic'],
			BPT.SetDynEnum: ['enum_dynamic'],
			# Special
			BPT.BLPointer: ['blptr_type'],
			BPT.Serialized: [],
		}[self]

	def check_info(self, prop_info: BLPropInfo) -> bool:
		"""Validate that a dictionary contains all required entries needed when creating a Blender property.

		Returns:
			True if the provided dictionary is guaranteed to result in a valid Blender property when used as keyword arguments in the `self.bl_prop` constructor.
		"""
		return all(
			required_info_key in prop_info for required_info_key in self.required_info
		)

	def parse_kwargs(  # noqa: PLR0915, PLR0912, C901
		self,
		obj_type: type,
		prop_info: BLPropInfo,
	) -> BLPropInfo:
		"""Parse the kwargs dictionary used to construct the Blender property.

		Parameters:
			obj_type: The exact object type that will be stored in the Blender property.
				**Generally** should be chosen such that `BLPropType.from_type(obj_type) == self`.
			prop_info: The property info.
				**Must** contain keys such that `required_info`

		Returns:
			Keyword arguments, which can be passed directly as to `self.bl_type` to construct a Blender property according to the `prop_info`.

			In total, creating a Blender property can be done simply using `self.bl_type(**parse_kwargs(...))`.
		"""
		BPT = BLPropType

		# Check Availability of Required Information
		## -> All required fields must be defined.
		if not self.check_info(prop_info):
			msg = f'{self} ({obj_type}): Required property attribute is missing from prop_info="{prop_info}"'
			raise ValueError(msg)

		# Define Information -> KWArg Getter
		def g_kwarg(name: str, force_key: str | None = None):
			key = force_key if force_key is not None else name
			return {key: prop_info[name]} if prop_info.get(name) is not None else {}

		# Encode Default Value
		if prop_info.get('default', Signal.CacheEmpty) is not Signal.CacheEmpty:
			encoded_default = {'default': self.encode(prop_info.get('default'))}
		else:
			encoded_default = {}

		# Assemble KWArgs
		kwargs = {}
		match self:
			case BPT.Bool if obj_type is bool:
				kwargs |= encoded_default

			case BPT.Int | BPT.IntVector | BPT.IntMatrix:
				kwargs |= encoded_default
				kwargs |= g_kwarg('abs_min')
				kwargs |= g_kwarg('abs_max')
				kwargs |= g_kwarg('soft_min')
				kwargs |= g_kwarg('soft_max')

			case BPT.Float | BPT.FloatVector | BPT.FloatMatrix:
				kwargs |= encoded_default
				kwargs |= g_kwarg('abs_min')
				kwargs |= g_kwarg('abs_max')
				kwargs |= g_kwarg('soft_min')
				kwargs |= g_kwarg('soft_max')
				kwargs |= g_kwarg('step')
				kwargs |= g_kwarg('precision')

			case BPT.Str if obj_type is str:
				kwargs |= encoded_default

				# Str: Secret
				if prop_info.get('str_secret'):
					kwargs |= {'subtype': 'PASSWORD', 'options': {'SKIP_SAVE'}}

				# Str: Search
				if prop_info.get('str_search'):
					kwargs |= g_kwarg('safe_str_cb', force_key='search')

			case BPT.Path if obj_type is Path:
				kwargs |= encoded_default

				# Path: File/Dir
				if prop_info.get('path_type'):
					kwargs |= {
						'subtype': (
							'FILE_PATH'
							if prop_info['path_type'] == 'file'
							else 'DIR_PATH'
						)
					}

			# Explicit Enums
			case BPT.SingleEnum:
				SubStrEnum = obj_type

				# Static | Dynamic Enum
				## -> Dynamic enums are responsible for respecting type.
				if prop_info.get('enum_dynamic'):
					kwargs |= g_kwarg('safe_enum_cb', force_key='items')
				else:
					kwargs |= encoded_default
					kwargs |= {
						'items': [
							## TODO: Parse __doc__ for item descs
							(
								str(value),
								SubStrEnum.to_name(value),
								SubStrEnum.to_name(value),
								SubStrEnum.to_icon(value),
								i,
							)
							for i, value in enumerate(list(obj_type))
						]
					}

			case BPT.SetEnum:
				SubStrEnum = typ.get_args(obj_type)[0]

				# Enum Set: Use ENUM_FLAG option.
				kwargs |= {'options': {'ENUM_FLAG'}}

				# Static | Dynamic Enum
				## -> Dynamic enums are responsible for respecting type.
				if prop_info.get('enum_dynamic'):
					kwargs |= g_kwarg('safe_enum_cb', force_key='items')
				else:
					kwargs |= encoded_default
					kwargs |= {
						'items': [
							## TODO: Parse __doc__ for item descs
							(
								str(value),
								SubStrEnum.to_name(value),
								SubStrEnum.to_name(value),
								SubStrEnum.to_icon(value),
								2**i,
							)
							for i, value in enumerate(list(SubStrEnum))
						]
					}

			# Anonymous Enums
			case BPT.SingleDynEnum:
				kwargs |= g_kwarg('safe_enum_cb', force_key='items')

			case BPT.SetDynEnum:
				kwargs |= g_kwarg('safe_enum_cb', force_key='items')

				# Enum Set: Use ENUM_FLAG option.
				kwargs |= {'options': {'ENUM_FLAG'}}

			# BLPointer
			case BPT.BLPointer:
				kwargs |= encoded_default

				# BLPointer: ID Type
				kwargs |= g_kwarg('blptr_type', force_key='type')

			# BLPointer
			case BPT.Serialized:
				kwargs |= encoded_default

		# Match Size
		## -> Matrices have inverted order to mitigate the Matrix Display Bug.
		size = self.parse_size(obj_type)
		if size is not None:
			if self in [BPT.BoolVector, BPT.IntVector, BPT.FloatVector]:
				kwargs |= {'size': size}
			if self in [BPT.BoolMatrix, BPT.IntMatrix, BPT.FloatMatrix]:
				kwargs |= {'size': size[::-1]}

		return kwargs

	####################
	# - Encode Value
	####################
	def encode(self, value: typ.Any) -> typ.Any:  # noqa: PLR0911
		"""Transform a value to a form that can be directly written to a Blender property.

		Parameters:
			value: A value which should be transformed into a form that can be written to the Blender property returned by `self.bl_type`.

		Returns:
			A value that can be written directly to the Blender property returned by `self.bl_type`.
		"""
		BPT = BLPropType
		match self:
			# Scalars: Coerce Losslessly
			## -> We choose to be very strict, except for float.is_integer() -> int
			case BPT.Bool if isinstance(value, bool):
				return value
			case BPT.Int if isinstance(value, int):
				return value
			case BPT.Int if isinstance(value, float) and value.is_integer():
				return int(value)
			case BPT.Float if isinstance(value, int | float):
				return float(value)

			# Vectors | Matrices: list()
			## -> We could use tuple(), but list() works just as fine when writing.
			## -> Later, we read back as tuple() to respect the type annotation.
			## -> Part of the workaround for the Matrix Display Bug happens here.
			case BPT.BoolVector | BPT.IntVector | BPT.FloatVector:
				return list(value)
			case BPT.BoolMatrix | BPT.IntMatrix | BPT.FloatMatrix:
				rows = len(value)
				cols = len(value[0])
				return (
					np.array(value, dtype=self.primitive_type)
					.flatten()
					.reshape([cols, rows])
				).tolist()

			# String
			## -> NOTE: This will happily encode StrEnums->str if an enum isn't requested.
			case BPT.Str if isinstance(value, str):
				return value

			# Path: Use Absolute-Resolved Stringified Path
			## -> TODO: Watch out for OS-dependence.
			case BPT.Path if isinstance(value, Path):
				return str(value.resolve())

			# Empty Enums
			## -> Coerce None to 'NONE', since 'NONE' is injected by convention.
			case (
				BPT.SingleEnum
				| BPT.SetEnum
				| BPT.SingleDynEnum
				| BPT.SetDynEnum
			) if value is None:
				return 'NONE'

			# Single Enum: Coerce to str
			## -> isinstance(StrEnum.Entry, str) always returns True; thus, a good sanity check.
			## -> Explicit/Dynamic both encode to str; only decode() coersion differentiates.
			case BPT.SingleEnum | BPT.SingleDynEnum if isinstance(value, str):
				return str(value)

			# Single Enum: Coerce to set[str]
			case BPT.SetEnum | BPT.SetDynEnum if isinstance(value, set):
				return {str(v) for v in value}

			# BLPointer: Don't Alter
			case BPT.BLPointer if value in BLIDStructs or value is None:
				return value

			# Serialized: Serialize To UTF-8
			## -> TODO: Check serializability
			case BPT.Serialized:
				return serialize.encode(value).decode('utf-8')

		msg = f'{self}: No encoder defined for argument {value}'
		raise NotImplementedError(msg)

	####################
	# - Decode Value
	####################
	def decode(self, raw_value: typ.Any, obj_type: type) -> typ.Any:  # noqa: PLR0911
		"""Transform a raw value from a form read directly from the Blender property returned by `self.bl_type`, to its intended value of approximate type `obj_type`.

		Notes:
			`obj_type` is only a hint - for example, `obj_type = enum.StrEnum` is an indicator for a dynamic enum.
			Its purpose is to guide ex. sizing and `StrEnum` coersion, not to guarantee a particular output type.

		Parameters:
			value: A value which should be transformed into a form that can be written to the Blender property returned by `self.bl_type`.

		Returns:
			A value that can be written directly to the Blender property returned by `self.bl_type`.
		"""
		BPT = BLPropType
		match self:
			# Scalars: Inverse Coerce (~Losslessly)
			## -> We choose to be very strict, except for float.is_integer() -> int
			case BPT.Bool if isinstance(raw_value, bool):
				return raw_value
			case BPT.Int if isinstance(raw_value, int):
				return raw_value
			case BPT.Int if isinstance(raw_value, float) and raw_value.is_integer():
				return int(raw_value)
			case BPT.Float if isinstance(raw_value, float):
				return float(raw_value)

			# Vectors | Matrices: tuple() to match declared type annotation.
			## -> Part of the workaround for the Matrix Display Bug happens here.
			case BPT.BoolVector | BPT.IntVector | BPT.FloatVector:
				return tuple(raw_value)
			case BPT.BoolMatrix | BPT.IntMatrix | BPT.FloatMatrix:
				rows, cols = self.parse_size(obj_type)
				return tuple(
					map(tuple, np.array(raw_value).flatten().reshape([rows, cols]))
				)

			# String
			## -> NOTE: This will happily decode StrEnums->str if an enum isn't requested.
			case BPT.Str if isinstance(raw_value, str):
				return raw_value

			# Path: Use 'Path(abspath(*))'
			## -> TODO: Watch out for OS-dependence.
			case BPT.Path if isinstance(raw_value, str):
				return Path(bpy.path.abspath(raw_value))

			# Empty Enums
			## -> Coerce 'NONE' to None, since 'NONE' is injected by convention.
			## -> Using coerced 'NONE' as guaranteed len=0 element is extremely helpful.
			case (
				BPT.SingleEnum
				| BPT.SetEnum
				| BPT.SingleDynEnum
				| BPT.SetDynEnum
			) if raw_value in ['NONE']:
				return None

			# Explicit Enum: Coerce to predefined StrEnum
			## -> This happens independent of whether there's a enum_cb.
			case BPT.SingleEnum if isinstance(raw_value, str):
				return obj_type(raw_value)
			case BPT.SetEnum if isinstance(raw_value, set):
				SubStrEnum = typ.get_args(obj_type)[0]
				return {SubStrEnum(v) for v in raw_value}

			## Dynamic Enums: Nothing to coerce to.
			## -> The critical distinction is that dynamic enums can't be coerced beyond str.
			## -> All dynamic enums have an enum_cb, but this is merely a symptom of ^.
			case BPT.SingleDynEnum if isinstance(raw_value, str):
				return raw_value
			case BPT.SetDynEnum if isinstance(raw_value, set):
				return raw_value

			# BLPointer
			## -> None is always valid when it comes to BLPointers.
			case BPT.BLPointer if raw_value in BLIDStructs or raw_value is None:
				return raw_value

			# Serialized: Deserialize the Argument
			case BPT.Serialized:
				return serialize.decode(obj_type, raw_value)

		msg = f'{self}: No decoder defined for argument {raw_value}'
		raise NotImplementedError(msg)

	####################
	# - Parse Type
	####################
	@staticmethod
	def from_type(obj_type: type) -> typ.Self:  # noqa: PLR0911, PLR0912, C901
		"""Select an appropriate `BLPropType` to store objects of the given type.

		Use of this method is especially handy when attempting to represent arbitrary, type-annotated objects using Blender properties.
		For example, the ability of the `BLPropType` to be displayed in a UI is prioritized as much as possible in making this decision.

		Parameters:
			obj_type: A type like `bool`, `str`, or custom classes.

		Returns:
			A `BLPropType` capable of storing any object of `obj_type`.
		"""
		BPT = BLPropType

		# Match Simple
		match obj_type:
			case builtins.bool:
				return BPT.Bool
			case builtins.int:
				return BPT.Int
			case builtins.float:
				return BPT.Float
			case builtins.str:
				return BPT.Str
			case pathlib.Path:
				return BPT.Path
			case enum.StrEnum:
				return BPT.SingleDynEnum
			case _:
				pass

		# Match Arrays
		## -> This deconstructs generic statements like ex. tuple[int, int]
		typ_origin = typ.get_origin(obj_type)
		typ_args = typ.get_args(obj_type)
		if typ_origin is tuple and len(typ_args) > 0:
			# Match Vectors
			## -> ONLY respect homogeneous types
			if all(T is bool for T in typ_args):
				return BPT.BoolVector
			if all(T is int for T in typ_args):
				return BPT.IntVector
			if all(T is float for T in typ_args):
				return BPT.FloatVector

			# Match Matrices
			## -> ONLY respect twice-nested homogeneous types
			## -> TODO: Explicitly require regularized shape, as in _parse_matrix_size
			typ_args_args = [typ_arg for T0 in typ_args for typ_arg in typ.get_args(T0)]
			if typ_args_args:
				if all(T is bool for T in typ_args_args):
					return BPT.BoolMatrix
				if all(T is int for T in typ_args_args):
					return BPT.IntMatrix
				if all(T is float for T in typ_args_args):
					return BPT.FloatMatrix

		# Match SetDynEnum
		## -> We can't do this in the match statement
		if obj_type == set[enum.StrEnum]:
			return BPT.SetDynEnum

		# Match Static Enums
		## -> Match Single w/Helper Function
		if _is_strenum(obj_type):
			return BPT.SingleEnum

		## -> Match Set w/Helper Function
		if typ_origin is set and len(typ_args) == 1 and _is_strenum(typ_args[0]):
			return BPT.SetEnum

		# Match BLPointers
		if obj_type in BLIDStructs:
			return BPT.BLPointer

		# Fallback: Serializable Object
		## -> TODO: Check serializability.
		return BPT.Serialized
