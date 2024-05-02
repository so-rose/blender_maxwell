"""Implements `PMLBoundCondNode`."""

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


class PMLBoundCondNode(base.MaxwellSimNode):
	r"""A "Perfectly Matched Layer" boundary condition, which is a theoretical medium that attempts to _perfectly_ absorb all outgoing waves, so as to represent "infinite space" in FDTD simulations.

	PML boundary conditions do so by inducing a **frequency-dependent attenuation** on all waves that are outside of the boundary, over the course of several layers.

	It is critical to note that a PML boundary can only absorb **propagating** waves.
	_Evanescent_ waves oscillating w/o any power flux, ex. close to structures, may actually be **amplified** by a PML boundary.
	This is the reasoning behind the $\frac{\lambda}{2}$-distance rule of thumb.

	For more theoretical details, please refer to the `tidy3d` resource: <https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.PML.html>

	Notes:
		**Ensure** that all simulation structures are $\approx \frac{\lambda}{2}$ from any PML boundary.

		This helps avoid the amplification of stray evanescent waves.

	Socket Sets:
		Simple: Only specify the number of PML layers.
			$12$ should cover the most common cases; $40$ should be extremely stable.
		Full: Specify the conductivity min/max that make up the PML, as well as the order of approximating polynomials.
			The meaning of the parameters are rooted in the mathematics that underlie the PML function - if that doesn't mean anything to you, then you should probably leave it alone!
			Since the value units are sim-relative, we've opted to show the scaling information in the node's UI, instead of coercing the values into any particular unit.
	"""

	node_type = ct.NodeType.PMLBoundCond
	bl_label = 'PML Bound Cond'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {
		'Layers': sockets.ExprSocketDef(
			shape=None,
			mathtype=spux.MathType.Integer,
			abs_min=1,
			default_value=12,
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
			'κ Order': sockets.ExprSocketDef(
				shape=None,
				mathtype=spux.MathType.Integer,
				abs_min=1,
				default_value=3,
			),
			'κ Range': sockets.ExprSocketDef(
				shape=(2,),
				mathtype=spux.MathType.Real,
				default_value=sp.Matrix([0, 1.5]),
				abs_min=0,
			),
			'α Order': sockets.ExprSocketDef(
				shape=None,
				mathtype=spux.MathType.Integer,
				abs_min=1,
				default_value=3,
			),
			'α Range': sockets.ExprSocketDef(
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
			for param in ['σ', 'κ', 'α']:
				col.label(text=param + ':')

			## RHS: Parameter Units
			col = split.column()
			for _ in range(3):
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
			'κ Order',
			'κ Range',
			'α Order',
			'α Range',
		},
		input_sockets_optional={
			'σ Order': True,
			'σ Range': True,
			'κ Order': True,
			'κ Range': True,
			'α Order': True,
			'α Range': True,
		},
	)
	def compute_pml_boundary_cond(self, props, input_sockets) -> td.PML:
		r"""Computes the PML boundary condition based on the active socket set.

		- **Simple**: Use `tidy3d`'s default parameters for defining the PML conductor (apart from number of layers).
		- **Full**: Use the user-defined $\sigma$, $\kappa$, and $\alpha$ parameters, specifically polynomial order and sim-relative min/max conductivity values.
		"""
		log.debug(
			'%s: Computing "%s" PML Boundary Condition (Input Sockets = %s)',
			self.sim_node_name,
			props['active_socket_set'],
			input_sockets,
		)

		# Simple PML
		if props['active_socket_set'] == 'Simple':
			return td.PML(num_layers=input_sockets['Layers'])

		# Full PML
		return td.PML(
			num_layers=input_sockets['Layers'],
			parameters=td.PMLParams(
				sigma_order=input_sockets['σ Order'],
				sigma_min=input_sockets['σ Range'][0],
				sigma_max=input_sockets['σ Range'][1],
				kappa_order=input_sockets['κ Order'],
				kappa_min=input_sockets['κ Range'][0],
				kappa_max=input_sockets['κ Range'][1],
				alpha_order=input_sockets['α Order'],
				alpha_min=input_sockets['α Range'][0],
				alpha_max=input_sockets['α Range'][1],
			),
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	PMLBoundCondNode,
]
BL_NODES = {ct.NodeType.PMLBoundCond: (ct.NodeCategory.MAXWELLSIM_BOUNDS)}
