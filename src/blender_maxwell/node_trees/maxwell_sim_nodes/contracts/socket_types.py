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

import enum

from blender_maxwell.utils import blender_type_enum


@blender_type_enum.append_cls_name_to_values
class SocketType(blender_type_enum.BlenderTypeEnum):
	Expr = enum.auto()

	# Base
	Any = enum.auto()
	Bool = enum.auto()
	String = enum.auto()
	FilePath = enum.auto()
	Color = enum.auto()

	# Blender
	BlenderMaterial = enum.auto()
	BlenderObject = enum.auto()
	BlenderCollection = enum.auto()

	BlenderImage = enum.auto()

	BlenderGeoNodes = enum.auto()
	BlenderText = enum.auto()

	# Maxwell
	MaxwellBoundConds = enum.auto()
	MaxwellBoundCond = enum.auto()

	MaxwellMedium = enum.auto()
	MaxwellMediumNonLinearity = enum.auto()

	MaxwellSource = enum.auto()
	MaxwellTemporalShape = enum.auto()

	MaxwellStructure = enum.auto()
	MaxwellMonitor = enum.auto()
	MaxwellMonitorData = enum.auto()

	MaxwellFDTDSim = enum.auto()
	MaxwellFDTDSimData = enum.auto()
	MaxwellSimDomain = enum.auto()
	MaxwellSimGrid = enum.auto()
	MaxwellSimGridAxis = enum.auto()

	# Tidy3D
	Tidy3DCloudTask = enum.auto()

	# Physical
	PhysicalUnitSystem = enum.auto()

	## Optical
	PhysicalPol = enum.auto()
