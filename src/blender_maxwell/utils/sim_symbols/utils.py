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

import sys
from fractions import Fraction

import sympy as sp

int_min = -(2**64)
int_max = 2**64
float_min = sys.float_info.min
float_max = sys.float_info.max


def unicode_superscript(n: int) -> str:
	"""Transform an integer into its unicode-based superscript character."""
	return ''.join(['⁰¹²³⁴⁵⁶⁷⁸⁹'[ord(c) - ord('0')] for c in str(n)])


def mk_interval(
	interval_finite: tuple[int | Fraction | float, int | Fraction | float],
	interval_inf: tuple[bool, bool],
	interval_closed: tuple[bool, bool],
) -> sp.Interval:
	"""Create a symbolic interval from the tuples (and unit) defining it."""
	return sp.Interval(
		start=(interval_finite[0] if not interval_inf[0] else -sp.oo),
		end=(interval_finite[1] if not interval_inf[1] else sp.oo),
		left_open=(True if interval_inf[0] else not interval_closed[0]),
		right_open=(True if interval_inf[1] else not interval_closed[1]),
	)
