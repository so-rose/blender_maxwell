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

from blender_maxwell.utils import logger

log = logger.get(__name__)

MANAGED_COLLECTION_NAME = 'BLMaxwell'
PREVIEW_COLLECTION_NAME = 'BLMaxwell Visible'


####################
# - Global Collection Handling
####################
def collection(collection_name: str, view_layer_exclude: bool) -> bpy.types.Collection:
	# Init the "Managed Collection"
	# Ensure Collection exists (and is in the Scene collection)
	if collection_name not in bpy.data.collections:
		collection = bpy.data.collections.new(collection_name)
		bpy.context.scene.collection.children.link(collection)
	else:
		collection = bpy.data.collections[collection_name]

	## Ensure synced View Layer exclusion
	if (
		layer_collection := bpy.context.view_layer.layer_collection.children[
			collection_name
		]
	).exclude != view_layer_exclude:
		layer_collection.exclude = view_layer_exclude

	return collection


def managed_collection() -> bpy.types.Collection:
	return collection(MANAGED_COLLECTION_NAME, view_layer_exclude=True)


def preview_collection() -> bpy.types.Collection:
	return collection(PREVIEW_COLLECTION_NAME, view_layer_exclude=False)
