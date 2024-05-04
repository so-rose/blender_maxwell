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

from . import eh_field_monitor, field_power_flux_monitor

# from . import epsilon_tensor_monitor
# from . import diffraction_monitor

BL_REGISTER = [
	*eh_field_monitor.BL_REGISTER,
	*field_power_flux_monitor.BL_REGISTER,
	# *epsilon_tensor_monitor.BL_REGISTER,
	# *diffraction_monitor.BL_REGISTER,
]
BL_NODES = {
	**eh_field_monitor.BL_NODES,
	**field_power_flux_monitor.BL_NODES,
	# **epsilon_tensor_monitor.BL_NODES,
	# **diffraction_monitor.BL_NODES,
}
