import enum

from ....utils.blender_type_enum import BlenderTypeEnum

class DataFlowKind(BlenderTypeEnum):
	"""Defines a shape/kind of data that may flow through a node tree.
	
	Since a node socket may define one of each, we can support several related kinds of data flow through the same node-graph infrastructure.
	
	Attributes:
		Value: A value usable without new data.
			- Basic types aka. float, int, list, string, etc. .
			- Exotic (immutable-ish) types aka. numpy array, KDTree, etc. .
			- A usable constructed object, ex. a `tidy3d.Box`.
			- Expressions (`sp.Expr`) that don't have unknown variables.
			- Lazy sequences aka. generators, with all data bound.
			
		LazyValue: An object which, when given new data, can make many values.
			- An `sp.Expr`, which might need `simplify`ing, `jax` JIT'ing, unit cancellations, variable substitutions, etc. before use.
			- Lazy objects, for which all parameters aren't yet known.
			- A computational graph aka. `aesara`, which may even need to be handled before 
		
		Capabilities: A `ValueCapability` object providing compatibility.
	
	# Value Data Flow
	Simply passing values is the simplest and easiest use case.
	
	This doesn't mean it's "dumb" - ex. a `sp.Expr` might, before use, have `simplify`, rewriting, unit cancellation, etc. run.
	All of this is okay, as long as there is no *introduction of new data* ex. variable substitutions.
	
	
	# Lazy Value Data Flow
	By passing (essentially) functions, one supports:
	- **Lightness**: While lazy values can be made expensive to construct, they will generally not be nearly as heavy to handle when trying to work with ex. operations on voxel arrays.
	- **Performance**: Parameterizing ex. `sp.Expr` with variables allows one to build very optimized functions, which can make ex. node graph updates very fast if the only operation run is the `jax` JIT'ed function (aka. GPU accelerated) generated from the final full expression.
	- **Numerical Stability**: Libraries like `aesara` build a computational graph, which can be automatically rewritten to avoid many obvious conditioning / cancellation errors.
	- **Lazy Output**: The goal of a node-graph may not be the definition of a single value, but rather, a parameterized expression for generating *many values* with known properties. This is especially interesting for use cases where one wishes to build an optimization step using nodes.
	
	
	# Capability Passing
	By being able to pass "capabilities" next to other kinds of values, nodes can quickly determine whether a given link is valid without having to actually compute it.
	
	
	# Lazy Parameter Value
	When using parameterized LazyValues, one may wish to independently pass parameter values through the graph, so they can be inserted into the final (cached) high-performance expression without.
	
	The advantage of using a different data flow would be changing this kind of value would ONLY invalidate lazy parameter value caches, which would allow an incredibly fast path of getting the value into the lazy expression for high-performance computation.
	
	Implementation TBD - though, ostensibly, one would have a "parameter" node which both would only provide a LazyValue (aka. a symbolic variable), but would also be able to provide a LazyParamValue, which would be a particular value of some kind (probably via the `value` of some other node socket).
	"""
	
	Value = enum.auto()
	LazyValue = enum.auto()
	Capabilities = enum.auto()
	
	LazyParamValue = enum.auto()
