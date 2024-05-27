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
import functools
import typing as typ
from types import MappingProxyType

import jax
import jaxtyping as jtyp

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger, sim_symbols

from .array import ArrayFlow
from .info import InfoFlow
from .lazy_range import RangeFlow
from .params import ParamsFlow

log = logger.get(__name__)

LazyFunction: typ.TypeAlias = typ.Callable[[typ.Any, ...], typ.Any]


@dataclasses.dataclass(frozen=True, kw_only=True)
class FuncFlow:
	r"""Defines a flow of data as incremental function composition.

	For specific math system usage instructions, please consult the documentation of relevant nodes.

	# Introduction
	When using nodes to do math, it becomes immediately obvious to express **flows of data as composed function chains**.
	Doing so has several advantages:

	- **Interactive**: Since no large-array math is being done, the UI can be designed to feel fast and snappy.
	- **Symbolic**: Since no numerical math is being done yet, we can choose to keep our input parameters as symbolic variables with no performance impact.
	- **Performant**: Since no operations are happening, the UI feels fast and snappy.

	## Strongly Related FlowKinds
	For doing math, `Func` relies on two other `FlowKind`s, which must run in parallel:

	- `FlowKind.Info`: Tracks the name, `spux.MathType`, unit (if any), length, and index coordinates for the raw data object produced by `Func`.
	- `FlowKind.Params`: Tracks the particular values of input parameters to the lazy function, each of which can also be symbolic.

	For more, please see the documentation for each.

	## Non-Mathematical Use
	Of course, there are many interesting uses of incremental function composition that aren't mathematical.

	For such cases, the usage is identical, but the complexity is lessened; for example, `Info` no longer effectively needs to flow in parallel.



	# Lazy Math: Theoretical Foundation
	This `FlowKind` is the critical component of a functional-inspired system for lazy multilinear math.
	Thus, it makes sense to describe the math system here.

	## `depth=0`: Root Function
	To start a composition chain, a function with no inputs must be defined as the "root", or "bottom".

	$$
		f_0:\ \ \ \ \biggl(
			\underbrace{a_1, a_2, ..., a_p}_{\texttt{args}},\ 
			\underbrace{
				\begin{bmatrix} k_1 \\ v_1\end{bmatrix},
				\begin{bmatrix} k_2 \\ v_2\end{bmatrix},
				...,
				\begin{bmatrix} k_q \\ v_q\end{bmatrix}
			}_{\texttt{kwargs}}
		\biggr) \to \text{output}_0
	$$

	In Python, such a construction would look like this:

	```python
	# Presume 'A0', 'KV0' contain only the args/kwargs for f_0
	## 'A0', 'KV0' are of length 'p' and 'q'
	def f_0(*args, **kwargs): ...

	lazy_func_0 = FuncFlow(
		func=f_0,
		func_args=[(a_i, type(a_i)) for a_i in A0],
		func_kwargs={k: v for k,v in KV0},
	)
	output_0 = lazy_func.func(*A0_computed, **KV0_computed)
	```

	## `depth>0`: Composition Chaining
	So far, so easy.
	Now, let's add a function that uses the result of $f_0$, without yet computing it.

	$$
		f_1:\ \ \ \ \biggl(
			f_0(...),\ \ 
			\underbrace{\{a_i\}_p^{p+r}}_{\texttt{args[p:]}},\ 
			\underbrace{\biggl\{
				\begin{bmatrix} k_i \\ v_i\end{bmatrix}
			\biggr\}_q^{q+s}}_{\texttt{kwargs[p:]}}
		\biggr) \to \text{output}_1
	$$

	Note:
	- $f_1$ must take the arguments of both $f_0$ and $f_1$.
	- The complexity is getting notationally complex; we already have to use `...` to represent "the last function's arguments".

	In other words, **there's suddenly a lot to manage**.
	Even worse, the bigger the $n$, the more complexity we must real with.

	This is where the Python version starts to show its purpose:

	```python
	# Presume 'A1', 'K1' contain only the args/kwarg names for f_1
	## 'A1', 'KV1' are therefore of length 'r' and 's'
	def f_1(output_0, *args, **kwargs): ...

	lazy_func_1 = lazy_func_0.compose_within(
		enclosing_func=f_1,
		enclosing_func_args=[(a_i, type(a_i)) for a_i in A1],
		enclosing_func_kwargs={k: type(v) for k,v in K1},
	)

	A_computed = A0_computed + A1_computed
	KW_computed = KV0_computed + KV1_computed
	output_1 = lazy_func_1.func(*A_computed, **KW_computed)
	```

	By using `Func`, we've guaranteed that even hugely deep $n$s won't ever look more complicated than this.

	## `max depth`: "Realization"
	So, we've composed a bunch of functions of functions of ...
	We've also tracked their arguments, either manually (as above), or with the help of a handy `ParamsFlow` object.

	But it'd be pointless to just compose away forever.
	We do actually need the data that they claim to compute now:

	```python
	# A_all and KW_all must be tracked on the side.
	output_n = lazy_func_n.func(*A_all, **KW_all)
	```

	Of course, this comes with enormous overhead.
	Aside from the function calls themselves (which can be non-trivial), we must also contend with the enormous inefficiency of performing array operations sequentially.

	That brings us to the killer feature of `FuncFlow`, and the motivating reason for doing any of this at all:
	
	```python
	output_n = lazy_func_n.func_jax(*A_all, **KW_all)
	```

	What happened was, **the entire pipeline** was compiled, optimized, and computed with bare-metal performance on either a CPU, GPU, or TPU.
	With the help of the `jax` library (and its underlying OpenXLA bytecode), all of that inefficiency has been optimized based on _what we're trying to do_, not _exactly how we're doing it_, in order to maximize the use of modern massively-parallel devices.

	See the documentation of `Func.func_jax()` for more information on this process.
	


	# Lazy Math: Practical Considerations
	By using nodes to express a lazily-composed chain of mathematical operations on tensor-like data, we strike a difficult balance between UX, flexibility, and performance.

	## UX
	UX is often more a matter of art/taste than science, so don't trust these philosophies too much - a lot of the analysis is entirely personal and subjective.

	The goal for our UX is to minimize the "frictions" that cause cascading, small-scale _user anxiety_.

	Especially of concern in a visual math system on large data volumes is **UX latency** - also known as **lag**.
	In particular, the most important facet to minimize is _emotional burden_ rather than quantitative milliseconds.
	Any repeated moment-to-moment friction can be very damaging to a user's ability to be productive in a piece of software.

	Unfortunately, in a node-based architecture, data must generally be computed one step at a time, whenever any part of it is needed, and it must do so before any feedback can be provided.
	In a math system like this, that data is presumed "big", and as such we're left with the unfortunate experience of even the most well-cached, high-performance operations causing _just about anything_ to **feel** like a highly unpleasant slog as soon as the data gets big enough.
	**This discomfort scales with the size of data**, by the way, which might just cause users to never even attempt working with the data volume that they actually need.

	For electrodynamic field analysis, it's not uncommon for toy examples to expend hundreds of megabytes of memory, all of which needs all manner of interesting things done to it.
	It can therefore be very easy to stumble across that feeling of "slogging through" any program that does real-world EM field analysis.
	This has consequences: The user tries fewer ideas, becomes more easily frustrated, and might ultimately accomplish less.

	Lazy evaluation allows _delaying_ a computation to a point in time where the user both expects and understands the time that the computation takes.
	For example, the user experience of pressing a button clearly marked with terminology like "load", "save", "compute", "run", seems to be paired to a greatly increased emotional tolerance towards the latency introduced by pressing that button (so long as it is only clickable when it works).
	To a lesser degree, attaching a node link also seems to have this property, though that tolerance seems to fall as proficiency with the node-based tool rises.
	As a more nuanced example, when lag occurs due to the computing an image-based plot based on live-computed math, then the visual feedback of _the plot actually changing_ seems to have a similar effect, not least because it's emotionally well-understood that detaching the `Viewer` node would also remove the lag.

	In short: Even if lazy evaluation didn't make any math faster, it will still _feel_ faster (to a point - raw performance obviously still matters).
	Without `FuncFlow`, the point of evaluation cannot be chosen at all, which is a huge issue for all the named reasons.
	With `FuncFlow`, better-chosen evaluation points can be chosen to cause the _user experience_ of high performance, simply because we were able to shift the exact same computation to a point in time where the user either understands or tolerates the delay better.

	## Flexibility
	Large-scale math is done on tensors, whether one knows (or likes!) it or not.
	To this end, the indexed arrays produced by `FuncFlow.func_jax` aren't quite sufficient for most operations we want to do:

	- **Naming**: What _is_ each axis?
		Unnamed index axes are sometimes easy to decode, but in general, names have an unexpectedly critical function when operating on arrays.
		Lack of names is a huge part of why perfectly elegant array math in ex. `MATLAB` or `numpy` can so easily feel so incredibly convoluted.
		_Sometimes arrays with named axes are called "structured arrays".

	- **Coordinates**: What do the indices of each axis really _mean_?
		For example, an array of $500$ by-wavelength observations of power (watts) can't be limited to between $200nm$ to $700nm$.
		But they can be limited to between index `23` to `298`.
		I'm **just supposed to know** that `23` means $200nm$, and that `298` indicates the observation just after $700nm$, and _hope_ that this is exact enough.

	Not only do we endeavor to track these, but we also introduce unit-awareness to the coordinates, and design the entire math system to visually communicate the state of arrays before/after every single computation, as well as only expose operations that this tracked data indicates possible.

	In practice, this happens in `FlowKind.Info`, which due to having its own `FlowKind` "lane" can be adjusted without triggering changes to (and therefore recompilation of) the `FlowKind.Func` chain.
	**Please consult the `InfoFlow` documentation for more**.

	## Performance
	All values introduced while processing are kept in a seperate `FlowKind` lane, with its own incremental caching: `FlowKind.Params`.

	It's a simple mechanism, but for the cost of introducing an extra `FlowKind` "lane", all of the values used to process data can be live-adjusted without the overhead of recompiling the entire `Func` every time anything changes.
	Moreover, values used to process data don't even have to be numbers yet: They can be expressions of symbolic variables, complete with units, which are only realized at the very end of the chain, by the node that absolutely cannot function without the actual numerical data.

	See the `ParamFlow` documentation for more information.



	# Conclusion
	There is, of course, a lot more to say about the math system in general.
	A few teasers of what nodes can do with this system:

	**Auto-Differentiation**: `jax.jit` isn't even really the killer feature of `jax`.
		`jax` can automatically differentiate `FuncFlow.func_jax` with respect to any input parameter, including for fwd/bck jacobians/hessians, with robust numerical stability.
		When used in 
	**Symbolic Interop**: Any `sympy` expression containing symbolic variables can be compiled, by `sympy`, into a `jax`-compatible function which takes 
		We make use of this in the `Expr` socket, enabling true symbolic math to be used in high-performance lazy `jax` computations.
	**Tidy3D Interop**: For some parameters of some simulation objects, `tidy3d` actually supports adjoint-driven differentiation _through the cloud simulation_.
		This enables our humble interface to implement fully functional **inverse design** of parameterized structures, using only nodes.

	But above all, we hope that this math system is fun, practical, and maybe even interesting.

	Attributes:
		func: The function that generates the represented value.
		func_args: The constrained identity of all positional arguments to the function.
		func_kwargs: The constrained identity of all keyword arguments to the function.
		supports_jax: Whether `self.func` can be compiled with JAX's JIT compiler.
			See the documentation of `self.func_jax()`.
	"""

	func: LazyFunction
	func_args: list[sim_symbols.SimSymbol] = dataclasses.field(default_factory=list)
	func_kwargs: dict[str, sim_symbols.SimSymbol] = dataclasses.field(
		default_factory=dict
	)
	supports_jax: bool = False

	####################
	# - Functions
	####################
	@functools.cached_property
	def func_jax(self) -> LazyFunction:
		"""Compile `self.func` into optimized XLA bytecode using `jax.jit`.

		Not all functions can be compiled like this by `jax`.
		A few critical criteria include:

		- **Only JAX Ops**: All operations performed within the function must be explicitly compatible with `jax`, which generally means only using functions in `jax.lax`, `jax.numpy`
		- **Known Shape**: The exact dimensions of the output, and of the inputs, must be known at `jit`-time.

		In return, one receives:

		- **Automatic Differentiation**: `jax` can robustly differentiate this function with respect to _any_ parameter.
			This includes Jacobians and Hessians, forwards and backwards, real or complex, all with good numerical stability.
			Since `tidy3d`'s simulator registers itself as `jax`-differentiable (using the adjoint method), this "autodiff" support can extend all the way from parameters in the simulation definition, to gradients of the simulation output.
			When using these gradients for optimization, one achieves what is called "inverse design", where the desired properties of the output fields are used to automatically select simulation input parameters.

		- **Performance**: XLA is a cross-industry project with the singular goal of providing a high-performance compilation target for data-driven programs.
			Published architects of OpenXLA include Alibaba, Amazon Web Services, AMD, Apple, Arm, Google, Intel, Meta, and NVIDIA.

		- **Device Agnosticism**: XLA bytecode runs not just on CPUs, but on massively parallel devices like GPUs and TPUs as well.
			This enables massive speedups, and greatly expands the amount of data that is practical to work with at one time.


		Notes:
			The property `self.supports_jax` manually tracks whether these criteria are satisfied.

			**As much as possible**, the _entirety of `blender_maxwell`_ is designed to maximize the ability to set `self.supports_jax = True` as often as possible.

			**However**, there are many cases where a lazily-evaluated value is desirable, but `jax` isn't supported.
			These include design space exploration, where any particular parameter might vary for the purpose of producing batched simulations.
			In these cases, trying to compile a `self.func_jax` will raise a `ValueError`.

		Returns:
			The `jit`-compiled function, ready to run on CPU, GPU, or XLA.

		Raises:
			ValueError: If `self.supports_jax` is `False`.

		References:
			JAX JIT: <https://jax.readthedocs.io/en/latest/jit-compilation.html>
			OpenXLA: <https://openxla.org/xla>
		"""
		if self.supports_jax:
			return jax.jit(self.func)

		msg = 'Can\'t express FuncFlow as JAX function (using jax.jit), since "self.supports_jax" is False'
		raise ValueError(msg)

	####################
	# - Realization
	####################
	def realize(
		self,
		params: ParamsFlow,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> typ.Self:
		if self.supports_jax:
			return self.func_jax(
				*params.scaled_func_args(symbol_values),
				**params.scaled_func_kwargs(symbol_values),
			)
		return self.func(
			*params.scaled_func_args(symbol_values),
			**params.scaled_func_kwargs(symbol_values),
		)

	def realize_as_data(
		self,
		info: InfoFlow,
		params: ParamsFlow,
		symbol_values: dict[sim_symbols.SimSymbol, spux.SympyExpr] = MappingProxyType(
			{}
		),
	) -> dict[sim_symbols.SimSymbol, jtyp.Inexact[jtyp.Array, '...']]:
		"""Realize as an ordered dictionary mapping each realized `self.dims` entry, with the last entry containing all output data as mapped from the `self.output`."""
		data = {}
		for dim, dim_idx in info.dims.items():
			# Continuous Index (*)
			## -> Continuous dimensions **must** be symbols in ParamsFlow.
			## -> ...Since the output data shape is parameterized by it.
			if info.has_idx_cont(dim):
				if dim in params.symbols:
					# Scalar Realization
					## -> Conform & cast the sympy expr to the dimension.
					if isinstance(symbol_values[dim], spux.SympyType):
						data |= {dim: dim.scale(symbol_values[dim])}

					# Array Realization
					## -> Scale the array to the dimension's unit & get values.
					if isinstance(symbol_values[dim], RangeFlow | ArrayFlow):
						data |= {
							dim: symbol_values[dim].rescale_to_unit(dim.unit).values
						}
				else:
					msg = f'ParamsFlow does not contain dimension symbol {dim} (info={info}, params={params})'
					raise RuntimeError(msg)

			# Discrete Index (Q|R)
			## -> Realize ArrayFlow|RangeFlow
			if info.has_idx_discrete(dim):
				data |= {dim: dim_idx.values}

			# Labelled Index (Z)
			## -> Passthrough the string labels.
			if info.has_idx_labels(dim):
				data |= {dim: dim_idx}

		return data | {info.output: self.realize(params, symbol_values=symbol_values)}

		# return {
		# dim: (
		# dim_idx
		# if info.has_idx_cont(dim) or info.has_idx_labels(dim)
		# else ??
		# )
		# for dim, dim_idx in self.dims
		# } | {info.output: output_data}

	####################
	# - Composition Operations
	####################
	def compose_within(
		self,
		enclosing_func: LazyFunction,
		enclosing_func_args: list[type] = (),
		enclosing_func_kwargs: dict[str, type] = MappingProxyType({}),
		supports_jax: bool = False,
	) -> typ.Self:
		"""Compose `self.func` within the given enclosing function, which itself takes arguments, and create a new `FuncFlow` to contain it.

		This is the fundamental operation used to "chain" functions together.

		Examples:
			Consider a simple composition based on two expressions:
			```python
			R = spux.MathType.Real
			C = spux.MathType.Complex
			x, y = sp.symbols('x y', real=True)

			# Prepare "Root" FuncFlow w/x,y args
			expr_root = 3*x + y**2 - 100
			expr_root_func = sp.lambdify([x, y], expr, 'jax')

			func_root = FuncFlow(func=expr_root_func, func_args=[R,R], supports_jax=True)

			# Compose "Enclosing" FuncFlow w/z arg
			r = sp.Symbol('z', real=True)
			z = sp.Symbol('z', complex=True)
			expr = 10*sp.re(z) / (z + r)
			expr_func = sp.lambdify([r, z], expr, 'jax')

			func = func_root.compose_within(enclosing_func=expr_func, enclosing_func_args=[C])

			# Compute 'expr_func(expr_root_func(10.0, -500.0), 1+8j)'
			f.func_jax(10.0, -500.0, 1+8j)
			```

			Using this function, it's easy to "keep adding" symbolic functions of any kind to the chain, without introducing extraneous complexity or compromising the ease of calling the final function.

		Returns:
			A lazy function that takes both the enclosed and enclosing arguments, and returns the value of the enclosing function (whose first argument is the output value of the enclosed function).
		"""
		return FuncFlow(
			func=lambda *args, **kwargs: enclosing_func(
				self.func(
					*list(args[: len(self.func_args)]),
					**{k: v for k, v in kwargs.items() if k in self.func_kwargs},
				),
				*args[len(self.func_args) :],
				**{k: v for k, v in kwargs.items() if k not in self.func_kwargs},
			),
			func_args=self.func_args + list(enclosing_func_args),
			func_kwargs=self.func_kwargs | dict(enclosing_func_kwargs),
			supports_jax=self.supports_jax and supports_jax,
		)

	def __or__(
		self,
		other: typ.Self,
	) -> typ.Self:
		"""Create a lazy function that takes all arguments of both lazy-function inputs, and itself promises to return a 2-tuple containing the outputs of both inputs.

		Generally, `self.func` produces a single array as output (when doing math, at least).
		But sometimes (as in the `OperateMathNode`), we need to perform a binary operation between two arrays, like say, $+$.
		Without realizing both `FuncFlow`s, it's not immediately obvious how one might accomplish this.

		This overloaded function of the `|` operator (used as `left | right`) solves that problem.
		A new `FuncFlow` is created, which takes the arguments of both inputs, and which produces a single output value: A 2-tuple, where each element if the output of each function.

		Examples:
			Consider this illustrative (pseudocode) example:
			```python
			# Presume a,b are values, and that A,B are their identifiers.
			func_1 = FuncFlow(func=compute_big_data_1, func_args=[A])
			func_2 = FuncFlow(func=compute_big_data_2, func_args=[B])

			f = (func_1 | func_2).compose_within(func=lambda D: D[0] + D[1])

			f.func(a, b)  ## Computes big_data_1 + big_data_2 @A=a, B=b
			```

			Because of `__or__` (the operator `|`), the difficult and non-obvious task of adding the outputs of these unrealized functions because quite simple.

		Notes:
			**Order matters**.
			`self` will be available in the new function's output as index `0`, while `other` will be available as index `1`.

			As with anything lazy-composition-y, it can seem a bit strange at first.
			When reading the source code, pay special attention to the way that `args` is sliced to segment the positional arguments.

		Returns:
			A lazy function that takes all arguments of both inputs, and returns a 2-tuple containing both output arguments.
		"""
		return FuncFlow(
			func=lambda *args, **kwargs: (
				self.func(
					*list(args[: len(self.func_args)]),
					**{k: v for k, v in kwargs.items() if k in self.func_kwargs},
				),
				other.func(
					*list(args[len(self.func_args) :]),
					**{k: v for k, v in kwargs.items() if k in other.func_kwargs},
				),
			),
			func_args=self.func_args + other.func_args,
			func_kwargs=self.func_kwargs | other.func_kwargs,
			supports_jax=self.supports_jax and other.supports_jax,
		)
