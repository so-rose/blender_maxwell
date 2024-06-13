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

"""Implements a `pydantic`-compatible field, `JaxArray`, which encapsulates a `jax.Array` in a serializable way."""

import base64
import io
import typing as typ

import jax
import jax.numpy as jnp
import numpy as np
import pydantic as pyd
from pydantic_core import core_schema as pyd_core_schema


####################
# - Simple JAX Array
####################
class _JaxArray:
	"""Annotated validator providing interoperability between `jax.Array` and `pydantic` models.

	Serializes to base64 bytes, for compatibility.
	"""

	@classmethod
	def __get_pydantic_core_schema__(
		cls, source_type: typ.Any, handler: pyd.GetCoreSchemaHandler
	) -> pyd_core_schema.CoreSchema:
		def validate_from_any(
			raw_array: bytes | jax.Array | list | tuple,
		) -> jax.Array:
			if isinstance(raw_array, np.ndarray):
				return jnp.array(raw_array)

			if isinstance(raw_array, jax.Array):
				return raw_array

			if isinstance(raw_array, bytes | str):
				with io.BytesIO() as memfile:
					memfile.write(base64.b64decode(raw_array.encode('utf-8')))
					memfile.seek(0)
					return jnp.load(memfile)

			if isinstance(raw_array, list | tuple):
				return jnp.array(raw_array)

			raise TypeError

		# def make_hashable(array: jax.Array) -> HashableJaxArray:
		# return HashableJaxArray(array)

		def serialize_to_bytes(array: jax.Array) -> bytes:
			with io.BytesIO() as memfile:
				jnp.save(memfile, array)
				return base64.b64encode(memfile.getvalue())

		jnp_array_schema = pyd_core_schema.chain_schema(
			[
				pyd_core_schema.no_info_plain_validator_function(validate_from_any),
				# pyd_core_schema.no_info_plain_validator_function(make_hashable),
				pyd_core_schema.is_instance_schema(jax.Array),
			]
		)
		return pyd_core_schema.json_or_python_schema(
			json_schema=jnp_array_schema,
			python_schema=jnp_array_schema,
			serialization=pyd_core_schema.plain_serializer_function_ser_schema(
				serialize_to_bytes
			),
		)


JaxArray = typ.Annotated[jax.Array, _JaxArray]


####################
# - Hashable JAX Array as-Bytes
####################
class _JaxArrayBytes:
	"""Annotated validator providing interoperability between `jax.Array` and `pydantic` models.

	Serializes to base64 bytes, for compatibility.
	"""

	@classmethod
	def __get_pydantic_core_schema__(
		cls, source_type: typ.Any, handler: pyd.GetCoreSchemaHandler
	) -> pyd_core_schema.CoreSchema:
		def validate_from_any(
			raw_array: bytes | jax.Array | list | tuple,
		) -> jax.Array:
			if isinstance(raw_array, np.ndarray):
				return jnp.array(raw_array)

			if isinstance(raw_array, jax.Array):
				return raw_array

			if isinstance(raw_array, bytes | str):
				with io.BytesIO() as memfile:
					memfile.write(base64.b64decode(raw_array.encode('utf-8')))
					memfile.seek(0)
					return jnp.load(memfile)

			if isinstance(raw_array, list | tuple):
				return jnp.array(raw_array)

			raise TypeError

		def to_bytes(array: jax.Array) -> bytes:
			with io.BytesIO() as memfile:
				jnp.save(memfile, array)
				return base64.b64encode(memfile.getvalue())

		jnp_array_bytes_schema = pyd_core_schema.chain_schema(
			[
				pyd_core_schema.no_info_plain_validator_function(validate_from_any),
				pyd_core_schema.no_info_plain_validator_function(to_bytes),
				pyd_core_schema.is_instance_schema(bytes),
			]
		)
		return pyd_core_schema.json_or_python_schema(
			json_schema=jnp_array_bytes_schema,
			python_schema=jnp_array_bytes_schema,
			serialization=pyd_core_schema.plain_serializer_function_ser_schema(bytes),
		)


JaxArrayBytes = typ.Annotated[bytes, _JaxArrayBytes]
