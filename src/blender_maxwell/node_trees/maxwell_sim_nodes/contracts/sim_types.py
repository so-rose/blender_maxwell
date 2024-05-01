"""Declares various simulation types for use by nodes and sockets."""

import enum
import typing as typ

import tidy3d as td


class BoundCondType(enum.StrEnum):
	r"""A type of boundary condition, applied to a half-axis of a simulation domain.

	Attributes:
		Pml: "Perfectly Matched Layer" models infinite free space.
			**Should be placed sufficiently far** (ex. $\frac{\lambda}{2}) from any active structures to mitigate divergence.
		Periodic: Denotes Bloch-basedrepetition
		Pec: "Perfect Electrical Conductor" models a surface that perfectly reflects electric fields.
		Pmc: "Perfect Magnetic Conductor" models a surface that perfectly reflects the magnetic fields.
	"""

	Pml = enum.auto()
	Periodic = enum.auto()
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
			BCT.Periodic: 'Periodic',
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
			BCT.Periodic: td.Periodic(),
		}[self]
