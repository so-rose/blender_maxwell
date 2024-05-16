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

"""Implements `BlochBoundCondNode`."""

import typing as typ

import bpy
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


class BlochBoundCondNode(base.MaxwellSimNode):
	r"""A boundary condition that declares an "infinitely repeating" window, by applying Bloch's theorem to accurately describe how a boundary would behave if it were interacting with an infinitely repeating simulation structure.

	# Theory
	In the simplest case, aka. a normal-incident plane wave, the symmetries of electromagnetic wave propagation behave exactly as expected: Copy-paste the wavevector, but at opposite corners, as part of the FDTD neighbor-cell-update.
	The moment this plane wave becomes angled, however, this "naive" method will cause **the phase of the periodically propagated fields to diverge from reality**.

	With a bit of hand-waving, this is a natural thing: Fundamentally, the distance from each point on an angled plane wave to the boundary must vary, and if the phase is distance-dependent, then the phase must vary across the boundary.

	Unfortunately, all of the explicitly-defined ways of describing how exactly to correct for this phenomenon depend on not only on what is being simulated, but on what is being studied.
	The good news is, there are options.

	## A Bloch of a Thing
	A physicist named Felix Bloch came up with a theorem to help constrain how "wave-like things in periodic stuff" can be thought about, and it looks like

	$$
		\psi(\mathbf{r}) = u(\mathbf{r}) \cdot \exp_{\mathbb{C}}(\mathbf{k} \cdot \mathbf{r})
	$$

	for:

	- $\psi$: A wave function (in general, satisfying the Schrödinger equation, but in this context, satisfying Maxwell's equations)
	- $\mathbf{r}$: A position in 3D space.
	- $\u$: Some periodic function mapping 3D space to a value. In this context, this might be a 3D function representing our simulation structures.
	- $\mathbf{k}$: The "Bloch vector", of which there is guaranteed to be at least one, **but of which there may be many**.

	At this point, it becomes interesting to note that pretty much _everything_ is, in fact, a "wave-like thing", so long as "the thing" is small enough.
	Many such "periodically structured things", which form entire fields of study, can indeed be modelled using this single function:

	- **Photonic Crystals**: The optical properties of many materials can be quite concisely encapsulated by placing regularly placed structures (of sub-wavelength size) within lattice-like structures.
	- **Phononic Crystals**: A class of metamaterial that can be parameterized and optimized for its acoustic properties, purely by analyzing its periodic behavior, with applications ranging from interesting acoustic devices to seismic modelling.

	## Modes of an Excited Nature
	For a choice of $u$ (representing the simulation structure), there may be _many continuous_ choices of $\mathbf{k}$ that satisfy the Bloch theorem.
	Similarly, for a particular choice of $\mathbf{k}$, there may be _several discrete_ particular solutions of the given wave function.

	Thus, we come full circle: **Fully encapsulating** the wave-interactions of a periodic structure requires knowing its behavior at **all valid wave vectors**.
	It is a sort of deeper truth, that any particular simulation of a unit cell cannot elicit the full story of _how a structure behaves_, since a particular choice of $\mathbf{k}$ must always inevitably be made as part of defining the simulation.

	## Designing Periodically
	With this insight in mind, we can now design simulations of periodic structures that properly account for the modalities imposed by particular $\mathbf{k}$ choices:

	- **Only Rely on Real Fields**: If only the real parts of the fields are important, then the choice of $\mathbf{k}$ might not matter.
		Remember, the symptom of needing to understand $\mathbf{k}$ is the phase-shift; if the phase-shift does not matter, then altering the Bloch vector won't change anything.
		**Be careful**, though, and make sure to validate that the Bloch vector truly doesn't change anything.
	- **Normal-Injected Plane Waves**: If fields generally only propagate in the normal direction, then again, choices of $\mathbf{k}$ might not matter.
		Again, phase-shifting due to periodic behavior mainly happens when propagation occurs at grazing angles.
		Again, **be careful**, and make sure to validate that ex. the Poynting vector truly isn't hitting the boundaries at too-grazing angles.
	- **Angularly Injected Plane Waves**: If the injected plane wave is known, then we can directly compute a reasonable Bloch vector from the angle and boundary-axis-projected size of the plane wave source.
		This selection of $\mathbf{k}$
	- **Brute-Force Bloch-Vector Sweep**: If the nature of a periodic structure needs to be uncovered, and there's no special constraints to rely on, then it would be rightfully tempting to just sweep over all $\mathbf{k}$s, and run a complete simluation for each.
		By going a step further, and plotting the energy of resonance frequencies noticed at each wave vector (just place point dipoles at random), one might stumble into a "band diagram" describing the possible energy states of electrons at each wave vector.

	In general, these form a very sensible starting point for how to select Bloch vectors for productive use in the simulation.

	NOTE: The Bloch vector is generally represented not as a vector, but as a single phase-shift per boundary axis unit length, mainly for convenience.

	## Further Reading
	- <https://optics.ansys.com/hc/en-us/articles/360041566614-Rectangular-Photonic-Crystal-Bandstructure>
	- <https://docs.flexcompute.com/projects/tidy3d/en/v2.1.0/notebooks/Bandstructure.html>
	- <https://en.wikipedia.org/wiki/Electronic_band_structure>
	- <https://en.wikipedia.org/wiki/Brillouin_zone>
	- <https://en.wikipedia.org/wiki/Bloch%27s_theorem>

	Notes:
		In the naive case, it is presumed that the choice of Bloch vector doesn't matter; therefore it is set to 0.

	Socket Sets:
		Naive: Specify a Bloch boundary condition where phase shift doesn't matter, and is thus set such that no phase-shift occurs.
			This is the simplest (and cheapest) mechanism, which merely copy-pastes propagating waves at opposing sides of the simulation.
			However, **this should not be used for angled plane waves**, as the phase-shift of a propagating angled plane wave **will be wrong**.
		Source-Derived: Derive a Bloch vector that will be generally correct for a directed source, within a particular choice of axis on a particular simulation domain.
			**Phase shift correctness is only guaranteed valid for the central frequency of the source**.
			Thus, a narrow-band source is strongly recommended.
		Bloch Vector: Specify a true Bloch boundary condition, including the **phase shift per unit length** (aka. the magnitude of the Bloch vector).
			While the most flexible, **the appropriate choice for this value source of this value depends entirely on what is being simulated**.
	"""

	node_type = ct.NodeType.BlochBoundCond
	bl_label = 'Bloch Bound Cond'

	####################
	# - Sockets
	####################
	input_socket_sets: typ.ClassVar = {
		'Naive': {},
		'Source-Derived': {
			'Angled Source': sockets.MaxwellSourceSocketDef(),
			## TODO: Constrain to gaussian beam, plane wafe, and tfsf
			'Sim Domain': sockets.MaxwellSimDomainSocketDef(),
		},
		'Manual': {
			'Bloch Vector': sockets.ExprSocketDef(),
		},
	}
	output_sockets: typ.ClassVar = {
		'BC': sockets.MaxwellBoundCondSocketDef(),
	}

	####################
	# - Properties
	####################
	valid_sim_axis: ct.SimSpaceAxis = bl_cache.BLField(ct.SimSpaceAxis.X)

	####################
	# - UI
	####################
	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		if self.active_socket_set == 'Source-Derived':
			layout.prop(self, self.blfields['valid_sim_axis'], expand=True)

	def draw_info(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		if self.active_socket_set == 'Manual':
			box = layout.box()
			row = box.row()
			row.alignment = 'CENTER'
			row.label(text='Interpretation')

			# Split
			split = box.split(factor=0.6, align=False)

			## LHS: Parameter Names
			col = split.column()
			col.alignment = 'RIGHT'
			col.label(text='Bloch Vec:')

			## RHS: Parameter Units
			col = split.column()
			col.label(text='2π/Δℓ')

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name={'active_socket_set', 'valid_sim_axis'},
		run_on_init=True,
		props={'active_socket_set', 'valid_sim_axis'},
	)
	def on_valid_sim_axis_changed(self, props):
		"""For the source-derived socket set, synchronized the output socket's axis compatibility with the axis onto which the Bloch vector is computed.

		The net result should be that invalid use of the Bloch boundary condition in a particular axis should be rejected.

		- **Source-Derived**: Since the Bloch vector is computed between the source and the axis that this boundary is applied to, the output socket must be altered to **only** declare compatibility with that axis.
		- **`*`**: Normalize the output socket axis validity to ensure that the boundary condition can be applied to any axis.
		"""
		if props['active_socket_set'] == 'Source-Derived':
			self.outputs['BC'].present_axes = {props['valid_sim_axis']}
			self.outputs['BC'].remove_invalidated_links()
		else:
			self.outputs['BC'].present_axes = {
				ct.SimSpaceAxis.X,
				ct.SimSpaceAxis.Y,
				ct.SimSpaceAxis.Z,
			}

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'BC',
		props={'active_socket_set', 'valid_sim_axis'},
		input_sockets={
			'Angled Source',
			'Sim Domain',
			'Bloch Vector',
		},
		input_sockets_optional={
			'Angled Source': True,
			'Sim Domain': True,
			'Bloch Vector': True,
		},
	)
	def compute_bloch_bound_cond(
		self, props, input_sockets
	) -> td.Periodic | td.BlochBoundary:
		r"""Computes the Bloch boundary condition.

		- **Naive**: Set the Bloch vector to 0 by returning a `td.Periodic`.
		- **Source-Derived**: Derive the Bloch vector from the source, simulation domain, and choice of axis.
			The Bloch boundary axis **must** be orthogonal to the source's injection axis.
		- **Manual**: Set the Bloch vector to the user-specified value.
		"""
		log.debug(
			'%s: Computing Bloch Boundary Condition (Socket Set = %s)',
			self.sim_node_name,
			props['active_socket_set'],
		)

		# Naive
		if props['active_socket_set'] == 'Naive':
			return td.Periodic()

		# Source-Derived
		if props['active_socket_set'] == 'Source-Derived':
			sim_domain = input_sockets['Sim Domain']
			valid_sim_axis = props['valid_sim_axis']

			has_sim_domain = not ct.FlowSignal.check(sim_domain)

			if has_sim_domain:
				return td.BlochBoundary.from_source(
					source=input_sockets['Angled Source'],
					domain_size=sim_domain['size'][valid_sim_axis.axis],
					axis=valid_sim_axis.axis,
					medium=sim_domain['medium'],
				)
			return ct.FlowSignal.FlowPending

		# Manual
		return td.BlochBoundary(bloch_vec=input_sockets['Bloch Vector'])


####################
# - Blender Registration
####################
BL_REGISTER = [
	BlochBoundCondNode,
]
BL_NODES = {ct.NodeType.BlochBoundCond: (ct.NodeCategory.MAXWELLSIM_BOUNDS)}
