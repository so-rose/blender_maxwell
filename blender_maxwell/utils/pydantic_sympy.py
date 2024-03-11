import typing as typ
import typing_extensions as typx

import pydantic as pyd
from pydantic_core import core_schema as pyd_core_schema
import sympy as sp
import sympy.physics.units as spu

from . import extra_sympy_units as spuex

####################
# - Missing Basics
####################
AllowedSympyExprs = sp.Expr | sp.MatrixBase
Complex = typx.Annotated[
	complex,
	pyd.GetPydanticSchema(
		lambda tp, handler: pyd_core_schema.no_info_after_validator_function(
			lambda x: x, handler(tp)
		)
	),
]

####################
# - Custom Pydantic Type for sp.Expr
####################
class _SympyExpr:
	@classmethod
	def __get_pydantic_core_schema__(
		cls,
		_source_type: AllowedSympyExprs,
		_handler: pyd.GetCoreSchemaHandler,
	) -> pyd_core_schema.CoreSchema:
		def validate_from_str(value: str) -> AllowedSympyExprs:
			if not isinstance(value, str):
				return value
			
			try:
				return sp.sympify(value)
			except ValueError as ex:
				msg = f"Value {value} is not a `sympify`able string"
				raise ValueError(msg) from ex
		
		def validate_from_expr(value: AllowedSympyExprs) -> AllowedSympyExprs:
			if not (
				isinstance(value, sp.Expr)
				or isinstance(value, sp.MatrixBase)
			):
				msg = f"Value {value} is not a `sympy` expression"
				raise ValueError(msg)
			
			return value
		
		sympy_expr_schema = pyd_core_schema.chain_schema([
			pyd_core_schema.no_info_plain_validator_function(validate_from_str),
			pyd_core_schema.no_info_plain_validator_function(validate_from_expr),
			pyd_core_schema.is_instance_schema(AllowedSympyExprs),
		])
		return pyd_core_schema.json_or_python_schema(
			json_schema=sympy_expr_schema,
			python_schema=sympy_expr_schema,
			serialization=pyd_core_schema.plain_serializer_function_ser_schema(
				lambda instance: str(instance)
			),
		)

####################
# - Configurable Expression Validation
####################
SympyExpr = typx.Annotated[
	AllowedSympyExprs,
	_SympyExpr,
]

def ConstrSympyExpr(
	# Feature Class
	allow_variables: bool = True,
	allow_units: bool = True,
	
	# Structure Class
	allowed_sets: set[typx.Literal[
		"integer", "rational", "real", "complex"
	]] | None = None,
	allowed_structures: set[typx.Literal[
		"scalar", "matrix"
	]] | None = None,
	
	# Element Class
	allowed_symbols: set[sp.Symbol] | None = None,
	allowed_units: set[spu.Quantity] | None = None,
	
	# Shape Class
	allowed_matrix_shapes: set[tuple[int, int]] | None = None,
):
	## See `sympy` predicates:
	## - <https://docs.sympy.org/latest/guides/assumptions.html#predicates>
	def validate_expr(expr: AllowedSympyExprs):
		if not (
			isinstance(expr, sp.Expr)
			or isinstance(expr, sp.MatrixBase),
		):
			## NOTE: Must match AllowedSympyExprs union elements.
			msg = f"expr '{expr}' is not an allowed Sympy expression ({AllowedSympyExprs})"
			raise ValueError(msg)
			
		msgs = set()
		
		# Validate Feature Class
		if (not allow_variables) and (len(expr.free_symbols) > 0):
			msgs.add(f"allow_variables={allow_variables} does not match expression {expr}.")
		if (not allow_units) and spuex.uses_units(expr):
			msgs.add(f"allow_units={allow_units} does not match expression {expr}.")
		
		# Validate Structure Class
		if allowed_sets and isinstance(expr, sp.Expr) and not any([
			{
				"integer": expr.is_integer,
				"rational": expr.is_rational,
				"real": expr.is_real,
				"complex": expr.is_complex,
			}[allowed_set]
			for allowed_set in allowed_sets
		]):
			msgs.add(f"allowed_sets={allowed_sets} does not match expression {expr} (remember to add assumptions to symbols, ex. `x = sp.Symbol('x', real=True))")
		if allowed_structures and not any([
			{
				"matrix": isinstance(expr, sp.MatrixBase),
			}[allowed_set]
			for allowed_set in allowed_structures
			if allowed_structures != "scalar"
		]):
			msgs.add(f"allowed_structures={allowed_structures} does not match expression {expr} (remember to add assumptions to symbols, ex. `x = sp.Symbol('x', real=True))")
		
		# Validate Element Class
		if allowed_symbols and expr.free_symbols.issubset(allowed_symbols):
			msgs.add(f"allowed_symbols={allowed_symbols} does not match expression {expr}")
		if allowed_units and spuex.get_units(expr).issubset(allowed_units):
			msgs.add(f"allowed_units={allowed_units} does not match expression {expr}")
		
		# Validate Shape Class
		if (
			allowed_matrix_shapes
			and isinstance(expr, sp.MatrixBase)
		) and not (expr.shape in allowed_matrix_shapes):
			msgs.add(f"allowed_matrix_shapes={allowed_matrix_shapes} does not match expression {expr} with shape {expr.shape}")
		
		# Error or Return
		if msgs: raise ValueError(str(msgs))
		return expr
	
	return typx.Annotated[
		AllowedSympyExprs,
		_SympyExpr,
		pyd.AfterValidator(validate_expr),
	]
