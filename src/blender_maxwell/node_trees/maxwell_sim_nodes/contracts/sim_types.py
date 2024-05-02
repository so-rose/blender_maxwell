"""Declares various simulation types for use by nodes and sockets."""

import enum
import typing as typ

import jax.numpy as jnp
import tidy3d as td


def manual_amp_time(self, time: float) -> complex:
	"""Copied implementation of `pulse.amp_time` for `tidy3d` temporal shapes, which replaces use of `numpy` with `jax.numpy` for `jit`-ability.

	Since the function is detached from the method, `self` is not implicitly available. It should be pre-defined from a real source time object using `functools.partial`, before `jax.jit`ing.

	## License
	**This function is directly copied from `tidy3d`**.
	As such, it should be considered available under the `tidy3d` license (as of writing, LGPL 2.1): <https://github.com/flexcompute/tidy3d/blob/develop/LICENSE>

	## Reference
	Permalink to GitHub source code: <https://github.com/flexcompute/tidy3d/blob/3ee34904eb6687a86a5fb3f4ed6d3295c228cd83/tidy3d/components/source.py#L143C1-L163C25>
	"""
	twidth = 1.0 / (2 * jnp.pi * self.fwidth)
	omega0 = 2 * jnp.pi * self.freq0
	time_shifted = time - self.offset * twidth

	offset = jnp.exp(1j * self.phase)
	oscillation = jnp.exp(-1j * omega0 * time)
	amp = jnp.exp(-(time_shifted**2) / 2 / twidth**2) * self.amplitude

	pulse_amp = offset * oscillation * amp

	# subtract out DC component
	if self.remove_dc_component:
		pulse_amp = pulse_amp * (1j + time_shifted / twidth**2 / omega0)
	else:
		# 1j to make it agree in large omega0 limit
		pulse_amp = pulse_amp * 1j

	return pulse_amp


## TODO: Sim Domain type, w/pydantic checks!


class SimSpaceAxis(enum.StrEnum):
	"""The axis labels of the global simulation coordinate system."""

	X = enum.auto()
	Y = enum.auto()
	Z = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		SSA = SimSpaceAxis
		return {
			SSA.X: 'x',
			SSA.Y: 'y',
			SSA.Z: 'z',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	@property
	def axis(self) -> int:
		"""Deduce the integer index of the axis.

		Returns:
			The integer index of the axis.
		"""
		SSA = SimSpaceAxis
		return {SSA.X: 0, SSA.Y: 1, SSA.Z: 2}[self]


class BoundCondType(enum.StrEnum):
	r"""A type of boundary condition, applied to a half-axis of a simulation domain.

	Attributes:
		Pml: "Perfectly Matched Layer" models infinite free space.
			**Should be placed sufficiently far** (ex. $\frac{\lambda}{2}) from any active structures to mitigate divergence.
		Periodic: Denotes naive Bloch boundaries (aka. periodic w/phase shift of 0).
		Pec: "Perfect Electrical Conductor" models a surface that perfectly reflects electric fields.
		Pmc: "Perfect Magnetic Conductor" models a surface that perfectly reflects the magnetic fields.
	"""

	Pml = enum.auto()
	NaiveBloch = enum.auto()
	Pec = enum.auto()
	Pmc = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		BCT = BoundCondType
		return {
			BCT.Pml: 'PML',
			BCT.Pec: 'PEC',
			BCT.Pmc: 'PMC',
			BCT.NaiveBloch: 'NaiveBloch',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	@property
	def tidy3d_boundary_edge(self) -> td.BoundaryEdge:
		"""Convert the boundary condition specifier to a corresponding, sensible `tidy3d` boundary edge.

		`td.BoundaryEdge` can be used to declare a half-axis in a `td.BoundarySpec`, which attaches directly to a simulation object.

		Returns:
			A sensible choice of `tidy3d` object representing the boundary condition.
		"""
		BCT = BoundCondType
		return {
			BCT.Pml: td.PML(),
			BCT.Pec: td.PECBoundary(),
			BCT.Pmc: td.PMCBoundary(),
			BCT.NaiveBloch: td.Periodic(),
		}[self]