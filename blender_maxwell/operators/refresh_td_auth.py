import bpy
from ..utils.auth_td_web import is_td_web_authed

class BlenderMaxwellRefreshTDAuth(bpy.types.Operator):
	bl_idname = "blender_maxwell.refresh_td_auth"
	bl_label = "Refresh Tidy3D Auth"
	bl_description = "Refresh the authentication of Tidy3D Web API"
	bl_options = {'REGISTER'}
	
	@classmethod
	def poll(cls, context):
		space = context.space_data
		return (
			space.type == 'NODE_EDITOR'
			and space.node_tree is not None
			and space.node_tree.bl_idname == "MaxwellSimTreeType"
		)

	def invoke(self, context, event):
		is_td_web_authed(force_check=True)
		return {'FINISHED'}

####################
# - Blender Registration
####################
BL_REGISTER = [
	BlenderMaxwellRefreshTDAuth,
]

BL_KMI_REGISTER = []
