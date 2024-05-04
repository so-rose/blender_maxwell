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

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

log = logger.get(__name__)

LazyFunction: typ.TypeAlias = typ.Callable[[typ.Any, ...], typ.Any]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyValueFuncFlow:
	r"""Wraps a composable function, providing useful information and operations.

	# Data Flow as Function Composition
	When using nodes to do math, it can be a good idea to express a **flow of data as the composition of functions**.

	Each node creates a new function, which uses the still-unknown (aka. **lazy**) output of the previous function to plan some calculations.
	Some new arguments may also be added, of course.

	## Root Function
	Of course, one needs to select a "bottom" function, which has no previous function as input.
	Thus, the first step is to define this **root function**:

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

	We'll express this simple snippet like so:

	```python
	# Presume 'A0', 'KV0' contain only the args/kwargs for f_0
	## 'A0', 'KV0' are of length 'p' and 'q'
	def f_0(*args, **kwargs): ...

	lazy_value_func_0 = LazyValueFuncFlow(
		func=f_0,
		func_args=[(a_i, type(a_i)) for a_i in A0],
		func_kwargs={k: v for k,v in KV0},
	)
	output_0 = lazy_value_func.func(*A0_computed, **KV0_computed)
	```

	So far so good.
	But of course, nothing interesting has really happened yet.

	## Composing Functions
	The key thing is the next step: The function that uses the result of $f_0$!

	$$
		f_1:\ \ \ \ \biggl(
			f_0(...),\ \ 
			\underbrace{\{a_i\}_p^{p+r}}_{\texttt{args[p:]}},\ 
			\underbrace{\biggl\{
				\begin{bmatrix} k_i \\ v_i\end{bmatrix}
			\biggr\}_q^{q+s}}_{\texttt{kwargs[p:]}}
		\biggr) \to \text{output}_1
	$$

	Notice that _$f_1$ needs the arguments of both $f_0$ and $f_1$_.
	Tracking arguments is already getting out of hand; we already have to use `...` to keep it readeable!

	But doing so with `LazyValueFunc` is not so complex:

	```python
	# Presume 'A1', 'K1' contain only the args/kwarg names for f_1
	## 'A1', 'KV1' are therefore of length 'r' and 's'
	def f_1(output_0, *args, **kwargs): ...

	lazy_value_func_1 = lazy_value_func_0.compose_within(
		enclosing_func=f_1,
		enclosing_func_args=[(a_i, type(a_i)) for a_i in A1],
		enclosing_func_kwargs={k: type(v) for k,v in K1},
	)

	A_computed = A0_computed + A1_computed
	KW_computed = KV0_computed + KV1_computed
	output_1 = lazy_value_func_1.func(*A_computed, **KW_computed)
	```

	We only need the arguments to $f_1$, and `LazyValueFunc` figures out how to make one function with enough arguments to call both.

	## Isn't Laying Functions Slow/Hard?
	Imagine that each function represents the action of a node, each of which performs expensive calculations on huge `numpy` arrays (**as one does when processing electromagnetic field data**).
	At the end, a node might run the entire procedure with all arguments:

	```python
	output_n = lazy_value_func_n.func(*A_all, **KW_all)
	```
	
	It's rough: Most non-trivial pipelines drown in the time/memory overhead of incremental `numpy` operations - individually fast, but collectively iffy.

	The killer feature of `LazyValueFuncFlow` is a sprinkle of black magic:
	
	```python
	func_n_jax = lazy_value_func_n.func_jax
	output_n = func_n_jax(*A_all, **KW_all)  ## Runs on your GPU
	```

	What happened was, **the entire pipeline** was compiled and optimized for high performance on not just your CPU, _but also (possibly) your GPU_.
	All the layered function calls and inefficient incremental processing is **transformed into a high-performance program**.

	Thank `jax` - specifically, `jax.jit` (https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit), which internally enables this magic with a single function call.
	
	## Other Considerations
	**Auto-Differentiation**: Incredibly, `jax.jit` isn't the killer feature of `jax`. The function that comes out of `LazyValueFuncFlow` can also be differentiated with `jax.grad` (read: high-performance Jacobians for optimizing input parameters).

	Though designed for machine learning, there's no reason other fields can't enjoy their inventions!

	**Impact of Independent Caching**: JIT'ing can be slow.
	That's why `LazyValueFuncFlow` has its own `FlowKind` "lane", which means that **only changes to the processing procedures will cause recompilation**.

	Generally, adjustable values that affect the output will flow via the `Param` "lane", which has its own incremental caching, and only meets the compiled function when it's "plugged in" for final evaluation.
	The effect is a feeling of snappiness and interactivity, even as the volume of data grows.

	Attributes:
		func: The function that the object encapsulates.
		bound_args: Arguments that will be packaged into function, which can't be later modifier.
		func_kwargs: Arguments to be specified by the user at the time of use.
		supports_jax: Whether the contained `self.function` can be compiled with JAX's JIT compiler.
	"""

	func: LazyFunction
	func_args: list[spux.MathType | spux.PhysicalType] = dataclasses.field(
		default_factory=list
	)
	func_kwargs: dict[str, spux.MathType | spux.PhysicalType] = dataclasses.field(
		default_factory=dict
	)
	supports_jax: bool = False

	# Merging
	def __or__(
		self,
		other: typ.Self,
	):
		return LazyValueFuncFlow(
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

	# Composition
	def compose_within(
		self,
		enclosing_func: LazyFunction,
		enclosing_func_args: list[type] = (),
		enclosing_func_kwargs: dict[str, type] = MappingProxyType({}),
		supports_jax: bool = False,
	) -> typ.Self:
		return LazyValueFuncFlow(
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

	@functools.cached_property
	def func_jax(self) -> LazyFunction:
		if self.supports_jax:
			return jax.jit(self.func)

		msg = 'Can\'t express LazyValueFuncFlow as JAX function (using jax.jit), since "self.supports_jax" is False'
		raise ValueError(msg)
