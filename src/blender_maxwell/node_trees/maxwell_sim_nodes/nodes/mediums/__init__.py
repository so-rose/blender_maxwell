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

# from . import pec_medium
# from . import isotropic_medium
# from . import anisotropic_medium
#
# from . import triple_sellmeier_medium
# from . import sellmeier_medium
# from . import drude_medium
# from . import drude_lorentz_medium
# from . import debye_medium
#
from . import library_medium, non_linearities, pole_residue_medium

BL_REGISTER = [
	*library_medium.BL_REGISTER,
	# *pec_medium.BL_REGISTER,
	# *isotropic_medium.BL_REGISTER,
	# *anisotropic_medium.BL_REGISTER,
	#
	# *triple_sellmeier_medium.BL_REGISTER,
	# *sellmeier_medium.BL_REGISTER,
	*pole_residue_medium.BL_REGISTER,
	# *drude_medium.BL_REGISTER,
	# *drude_lorentz_medium.BL_REGISTER,
	# *debye_medium.BL_REGISTER,
	#
	*non_linearities.BL_REGISTER,
]
BL_NODES = {
	**library_medium.BL_NODES,
	# **pec_medium.BL_NODES,
	# **isotropic_medium.BL_NODES,
	# **anisotropic_medium.BL_NODES,
	#
	# **triple_sellmeier_medium.BL_NODES,
	# **sellmeier_medium.BL_NODES,
	**pole_residue_medium.BL_NODES,
	# **drude_medium.BL_NODES,
	# **drude_lorentz_medium.BL_NODES,
	# **debye_medium.BL_NODES,
	#
	**non_linearities.BL_NODES,
}
