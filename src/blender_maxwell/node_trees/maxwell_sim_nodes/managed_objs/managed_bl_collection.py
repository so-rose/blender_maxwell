
import bpy

from ....utils import logger

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
