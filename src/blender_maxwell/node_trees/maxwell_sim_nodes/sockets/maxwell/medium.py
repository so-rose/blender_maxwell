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

import bpy
import scipy as sc
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from blender_maxwell.utils import bl_cache, logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from ... import contracts as ct
from .. import base

log = logger.get(__name__)

_VAC_SPEED_OF_LIGHT = sc.constants.speed_of_light * spu.meter / spu.second
_FIXED_WL = 500 * spu.nm
FIXED_FREQ = spux.scale_to_unit_system(
	spu.convert_to(
		_VAC_SPEED_OF_LIGHT / _FIXED_WL,
		spu.hertz,
	),
	ct.UNITS_TIDY3D,
)


class MaxwellMediumBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellMedium
	bl_label = 'Maxwell Medium'

	####################
	# - Properties
	####################
	eps_rel: tuple[float, float] = bl_cache.BLField((1.0, 0.0), float_prec=2)

	####################
	# - FlowKinds
	####################
	@bl_cache.cached_bl_property(depends_on={'eps_rel'})
	def value(self) -> td.Medium:
		eps_r_re = self.eps_rel[0]
		# conductivity = FIXED_FREQ * self.eps_rel[1]  ## TODO: Um?

		return td.Medium(
			permittivity=eps_r_re,
			# conductivity=conductivity,
		)

	@value.setter
	def value(self, eps_rel: tuple[float, float]) -> None:
		self.eps_rel = eps_rel

	@bl_cache.cached_bl_property()
	def lazy_func(self) -> ct.FuncFlow:
		return ct.FuncFlow(
			func=lambda eps_r_re, eps_r_im: td.Medium(
				permittivity=eps_r_re,
				# conductivity=FIXED_FREQ * eps_r_im,
			),
			func_args=[sim_symbols.rel_eps_re(None), sim_symbols.rel_eps_im(None)],
		)

	@bl_cache.cached_bl_property(depends_on={'eps_rel'})
	def params(self) -> ct.FuncFlow:
		return ct.ParamsFlow(
			func_args=[sp.S(self.eps_rel[0]), sp.S(self.eps_rel[1])],
		)
		return ct.ParamsFlow()

	####################
	# - UI
	####################
	def draw_value(self, col: bpy.types.UILayout) -> None:
		split = col.split(factor=0.25, align=False)

		_col = split.column(align=True)
		_col.label(text='εᵣ')

		_col = split.column(align=True)
		_col.prop(self, self.blfields['eps_rel'], text='')


####################
# - Socket Configuration
####################
class MaxwellMediumSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellMedium

	default_permittivity_real: float = 1.0
	default_permittivity_imag: float = 0.0

	def init(self, bl_socket: MaxwellMediumBLSocket) -> None:
		bl_socket.eps_rel = (
			self.default_permittivity_real,
			self.default_permittivity_imag,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellMediumBLSocket,
]
