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

from .socket_types import SocketType as ST

## TODO: Don't just presume sRGB.
SOCKET_COLORS = {
	# Basic
	ST.Any: (0.9, 0.9, 0.9, 1.0),  # Light Grey
	ST.Bool: (0.7, 0.7, 0.7, 1.0),  # Medium Light Grey
	ST.String: (0.7, 0.7, 0.7, 1.0),  # Medium Light Grey
	ST.FilePath: (0.6, 0.6, 0.6, 1.0),  # Medium Grey
	ST.Expr: (0.5, 0.5, 0.5, 1.0),  # Medium Grey
	# Physical
	ST.PhysicalUnitSystem: (1.0, 0.5, 0.5, 1.0),  # Light Red
	ST.PhysicalPol: (0.5, 0.4, 0.2, 1.0),  # Dark Orange
	# Blender
	ST.BlenderMaterial: (0.8, 0.6, 1.0, 1.0),  # Lighter Purple
	ST.BlenderObject: (0.7, 0.5, 1.0, 1.0),  # Light Purple
	ST.BlenderCollection: (0.6, 0.45, 0.9, 1.0),  # Medium Light Purple
	ST.BlenderImage: (0.5, 0.4, 0.8, 1.0),  # Medium Purple
	ST.BlenderGeoNodes: (0.3, 0.3, 0.6, 1.0),  # Dark Purple
	ST.BlenderText: (0.5, 0.5, 0.75, 1.0),  # Light Lavender
	# Maxwell
	ST.MaxwellSource: (1.0, 1.0, 0.5, 1.0),  # Light Yellow
	ST.MaxwellTemporalShape: (0.9, 0.9, 0.45, 1.0),  # Medium Light Yellow
	ST.MaxwellMedium: (0.8, 0.8, 0.4, 1.0),  # Medium Yellow
	ST.MaxwellMediumNonLinearity: (0.7, 0.7, 0.35, 1.0),  # Medium Dark Yellow
	ST.MaxwellStructure: (0.6, 0.6, 0.3, 1.0),  # Dark Yellow
	ST.MaxwellBoundConds: (0.9, 0.8, 0.5, 1.0),  # Light Gold
	ST.MaxwellBoundCond: (0.8, 0.7, 0.45, 1.0),  # Medium Light Gold
	ST.MaxwellMonitor: (0.7, 0.6, 0.4, 1.0),  # Medium Gold
	ST.MaxwellMonitorData: (0.7, 0.6, 0.4, 1.0),  # Medium Gold
	ST.MaxwellFDTDSim: (0.6, 0.5, 0.35, 1.0),  # Medium Dark Gold
	ST.MaxwellFDTDSimData: (0.6, 0.5, 0.35, 1.0),  # Medium Dark Gold
	ST.MaxwellSimGrid: (0.5, 0.4, 0.3, 1.0),  # Dark Gold
	ST.MaxwellSimGridAxis: (0.4, 0.3, 0.25, 1.0),  # Darkest Gold
	ST.MaxwellSimDomain: (0.4, 0.3, 0.25, 1.0),  # Darkest Gold
	# Tidy3D
	ST.Tidy3DCloudTask: (0.4, 0.3, 0.25, 1.0),  # Darkest Gold
}
