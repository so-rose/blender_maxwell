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

"""Declares a useful, flexible symbolic representation."""

from .common import (
	CommonSimSymbol,
	ang_phi,
	ang_r,
	ang_theta,
	diff_order_x,
	diff_order_y,
	dir_x,
	dir_y,
	dir_z,
	field_e,
	field_ex,
	field_ey,
	field_ez,
	field_h,
	field_hx,
	field_hy,
	field_hz,
	flux,
	freq,
	idx,
	rel_eps_im,
	rel_eps_re,
	sim_axis_idx,
	space_x,
	space_y,
	space_z,
	t,
	wl,
)
from .name import SimSymbolName
from .sim_symbol import SimSymbol
from .utils import (
	float_max,
	float_min,
	int_max,
	int_min,
	mk_interval,
	unicode_superscript,
)

__all__ = [
	'CommonSimSymbol',
	'idx',
	'rel_eps_im',
	'rel_eps_re',
	'sim_axis_idx',
	't',
	'wl',
	'freq',
	'space_x',
	'space_y',
	'space_z',
	'dir_x',
	'dir_y',
	'dir_z',
	'ang_r',
	'ang_theta',
	'ang_phi',
	'field_e',
	'field_ex',
	'field_ey',
	'field_ez',
	'field_h',
	'field_hx',
	'field_hy',
	'field_hz',
	'flux',
	'diff_order_x',
	'diff_order_y',
	'SimSymbolName'
	'SimSymbol'
	'float_max'
	'float_min'
	'int_max'
	'int_min'
	'mk_interval'
	'unicode_superscript',
]
