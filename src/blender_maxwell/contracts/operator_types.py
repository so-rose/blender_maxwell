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

"""Defines Operator Types as an enum, making it easy for any part of the addon to refer to any operator."""

import enum

from ..nodeps.utils import blender_type_enum
from .addon import NAME as ADDON_NAME


@blender_type_enum.prefix_values_with(f'{ADDON_NAME}.')
class OperatorType(enum.StrEnum):
	"""Identifiers for addon-defined `bpy.types.Operator`."""

	InstallPyDeps = enum.auto()
	UninstallPyDeps = enum.auto()
	ManagePyDeps = enum.auto()

	ConnectViewerNode = enum.auto()

	GeoNodesToStructureNode = enum.auto()

	# Socket: Tidy3DCloudTask
	SocketCloudAuthenticate = enum.auto()
	SocketReloadCloudFolderList = enum.auto()

	# Node: Tidy3DWebImporter
	NodeLoadCloudSim = enum.auto()

	# Node: Tidy3DWebExporter
	NodeRecomputeSimInfo = enum.auto()
	NodeUploadSimulation = enum.auto()
	NodeReleaseUploadedTask = enum.auto()
	NodeRunSimulation = enum.auto()
	NodeReloadTrackedTask = enum.auto()
