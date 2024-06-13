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

"""Implements a `pydantic`-compatible field, `FrozenDict`, which encapsulates a `frozendict` in a serializable way, with semantics identical to `dict`."""

import typing as typ

import pydantic as pyd
from frozendict import deepfreeze, frozendict
from pydantic_core import core_schema as pyd_core_schema


class _PydanticFrozenDictAnnotation:
	"""Annotated validator providing interoperability between `frozendict` and `pydantic` models.

	Semantics are almost identical to `dict`, except for a chained conversion to `frozendict`.
	"""

	@classmethod
	def __get_pydantic_core_schema__(
		cls, source_type: typ.Any, handler: pyd.GetCoreSchemaHandler
	) -> pyd_core_schema.CoreSchema:
		def validate_from_dict(d: dict | frozendict) -> frozendict:
			return frozendict(d)

		frozendict_schema = pyd_core_schema.chain_schema(
			[
				handler.generate_schema(dict[*typ.get_args(source_type)]),
				pyd_core_schema.no_info_plain_validator_function(validate_from_dict),
				pyd_core_schema.is_instance_schema(frozendict),
			]
		)
		return pyd_core_schema.json_or_python_schema(
			json_schema=frozendict_schema,
			python_schema=frozendict_schema,
			serialization=pyd_core_schema.plain_serializer_function_ser_schema(dict),
		)


_K = typ.TypeVar('_K')
_V = typ.TypeVar('_V')
FrozenDict = typ.Annotated[frozendict[_K, _V], _PydanticFrozenDictAnnotation]

__all__ = ['deepfreeze', 'frozendict', 'FrozenDict']
