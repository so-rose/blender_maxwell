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

"""Declares many useful primitives to greatly simplify working with `sympy` in the context of a unit-aware system."""

from .math_type import MathType
from .number_size import NumberSize1D, NumberSize2D
from .parse_cast import parse_shape, pretty_symbol, sp_to_str, sympy_to_python
from .physical_type import Dims, PhysicalType
from .sympy_expr import (
	ComplexNumber,
	ComplexSymbol,
	ConstrSympyExpr,
	IntNumber,
	IntSymbol,
	Number,
	PhysicalComplexNumber,
	PhysicalNumber,
	PhysicalRealNumber,
	RationalSymbol,
	Real3DVector,
	RealNumber,
	RealSymbol,
	ScalarUnitlessComplexExpr,
	ScalarUnitlessRealExpr,
	Symbol,
	SympyExpr,
	Unit,
	UnitDimension,
)
from .sympy_type import SympyType
from .unit_analysis import (
	compare_unit_dims,
	compare_units_by_unit_dims,
	convert_to_unit,
	get_units,
	scale_to_unit,
	scaling_factor,
	strip_units,
	unit_dim_to_unit_dim_deps,
	unit_str_to_unit,
	unit_to_unit_dim_deps,
	uses_units,
)
from .unit_system_analysis import (
	convert_to_unit_system,
	scale_to_unit_system,
	strip_unit_system,
)
from .unit_systems import UNITS_SI, UnitSystem
from .units import (
	UNIT_BY_SYMBOL,
	UNIT_TO_1,
	EHz,
	GHz,
	KHz,
	MHz,
	PHz,
	THz,
	exahertz,
	femtometer,
	femtosecond,
	fm,
	fs,
	gigahertz,
	hectopascal,
	hPa,
	kilohertz,
	lm,
	lumen,
	mbar,
	megahertz,
	micronewton,
	millibar,
	millinewton,
	mN,
	nanonewton,
	nN,
	petahertz,
	terahertz,
	uN,
)

__all__ = [
	'MathType',
	'NumberSize1D',
	'NumberSize2D',
	'parse_shape',
	'pretty_symbol',
	'sp_to_str',
	'sympy_to_python',
	'Dims',
	'PhysicalType',
	'ComplexNumber',
	'ComplexSymbol',
	'ConstrSympyExpr',
	'IntNumber',
	'IntSymbol',
	'Number',
	'PhysicalComplexNumber',
	'PhysicalNumber',
	'PhysicalRealNumber',
	'RationalSymbol',
	'Real3DVector',
	'RealNumber',
	'RealSymbol',
	'ScalarUnitlessComplexExpr',
	'ScalarUnitlessRealExpr',
	'Symbol',
	'SympyExpr',
	'Unit',
	'UnitDimension',
	'SympyType',
	'compare_unit_dims',
	'compare_units_by_unit_dims',
	'convert_to_unit',
	'get_units',
	'scale_to_unit',
	'scaling_factor',
	'strip_units',
	'unit_dim_to_unit_dim_deps',
	'unit_str_to_unit',
	'unit_to_unit_dim_deps',
	'uses_units',
	'strip_unit_system',
	'UNITS_SI',
	'UnitSystem',
	'convert_to_unit_system',
	'scale_to_unit_system',
	'UNIT_BY_SYMBOL',
	'UNIT_TO_1',
	'EHz',
	'GHz',
	'KHz',
	'MHz',
	'PHz',
	'THz',
	'exahertz',
	'femtometer',
	'femtosecond',
	'fm',
	'fs',
	'gigahertz',
	'hectopascal',
	'hPa',
	'kilohertz',
	'lm',
	'lumen',
	'mbar',
	'megahertz',
	'micronewton',
	'millibar',
	'millinewton',
	'mN',
	'nanonewton',
	'nN',
	'petahertz',
	'terahertz',
	'uN',
]
