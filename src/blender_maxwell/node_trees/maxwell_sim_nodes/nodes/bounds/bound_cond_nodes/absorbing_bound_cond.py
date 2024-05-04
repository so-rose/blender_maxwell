"""Implements `AdiabAbsorbBoundCondNode`."""

import typing as typ

import bpy
import sympy as sp
import tidy3d as td

from blender_maxwell.utils import extra_sympy_units as spux
from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class AdiabAbsorbBoundCondNode(base.MaxwellSimNode):
	r"""A boundary condition that generically (adiabatically) absorbs outgoing energy, by gradually ramping up the strength of the conductor over many layers, until a final PEC layer.

	Compared to PML, this boundary is more computationally expensive, and may result in higher reflectivity (and thus lower accuracy).
	The general reason to use it is **to fix divergence in cases where dispersive materials intersect the simulation boundary**.

	For more theoretical details, please refer to the `tidy3d` resource: <https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Absorber.html>

	Notes:
		**Ensure** that all simulation structures are $\approx \frac{\lambda}{2}$ from any PML boundary.

		This helps avoid the amplification of stray evanescent waves.

	Socket Sets:
		Simple: Only specify the number of absorption layers.
			$40$ should generally be sufficient, but in the case of divergence issues, bumping up the number of layers should be the go-to remedy to try.
		Full: Specify the conductivity min/max that makes up the absorption up the PML, as well as the order of the polynomial used to scale the effect through the layers.
			You should probably leave this alone.
			Since the value units are sim-relative, we've opted to show the scaling information in the node's UI, instead of coercing the values into any particular unit.
	"""

	node_type = ct.NodeType.AdiabAbsorbBoundCond
	bl_label = 'Absorber Bound Cond'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Layers': sockets.ExprSocketDef(
			shape=None,
			mathtype=spux.MathType.Integer,
			abs_min=1,
			default_value=40,
		),
	}
	input_socket_sets: typ.ClassVar = {
		'Simple': {},
		'Full': {
			'σ Order': sockets.ExprSocketDef(
				shape=None,
				mathtype=spux.MathType.Integer,
				abs_min=1,
				default_value=3,
			),
			'σ Range': sockets.ExprSocketDef(
				shape=(2,),
				mathtype=spux.MathType.Real,
				default_value=sp.Matrix([0, 1.5]),
				abs_min=0,
			),
		},
	}
	output_sockets: typ.ClassVar = {
		'BC': sockets.MaxwellBoundCondSocketDef(),
	}

	####################
	# - UI
	####################
	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		if self.active_socket_set == 'Full':
			box = layout.box()
			row = box.row()
			row.alignment = 'CENTER'
			row.label(text='Parameter Scale')

			# Split
			split = box.split(factor=0.4, align=False)

			## LHS: Parameter Names
			col = split.column()
			col.alignment = 'RIGHT'
			col.label(text='σ:')

			## RHS: Parameter Units
			col = split.column()
			col.label(text='2ε₀/Δt')

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'BC',
		props={'active_socket_set'},
		input_sockets={
			'Layers',
			'σ Order',
			'σ Range',
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
		},
	)
	def compute_adiab_absorber_bound_cond(self, props, input_sockets) -> td.Absorber:
		r"""Computes the adiabatic absorber boundary condition based on the active socket set.

		- **Simple**: Use `tidy3d`'s default parameters for defining the absorber parameters (apart from number of layers).
		- **Full**: Use the user-defined $\sigma$ parameters, specifically polynomial order and sim-relative min/max conductivity values.
		"""
		log.debug(
			'%s: Computing "%s" Adiabatic Absorber Boundary Condition (Input Sockets = %s)',
			self.sim_node_name,
			props['active_socket_set'],
			input_sockets,
		)

		# Simple PML
		if props['active_socket_set'] == 'Simple':
			return td.Absorber(num_layers=input_sockets['Layers'])

		# Full PML
		return td.Absorber(
			num_layers=input_sockets['Layers'],
			parameters=td.AbsorberParams(
				sigma_order=input_sockets['σ Order'],
				sigma_min=input_sockets['σ Range'][0],
				sigma_max=input_sockets['σ Range'][1],
			),
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	AdiabAbsorbBoundCondNode,
]
BL_NODES = {ct.NodeType.AdiabAbsorbBoundCond: (ct.NodeCategory.MAXWELLSIM_BOUNDS)}
