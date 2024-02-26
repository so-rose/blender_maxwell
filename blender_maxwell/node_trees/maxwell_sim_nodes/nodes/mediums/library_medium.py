import bpy
import tidy3d as td
import sympy as sp
import sympy.physics.units as spu
import numpy as np
import scipy as sc

from .....utils import extra_sympy_units as spuex
from ... import contracts
from ... import sockets
from .. import base

class ExperimentOperator00(bpy.types.Operator):
	bl_idname = "blender_maxwell.experiment_operator_00"
	bl_label = "exp"

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		node = context.node
		node.invoke_matplotlib_and_update_image()
		return {'FINISHED'}

class LibraryMediumNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.LibraryMedium
	
	bl_label = "Library Medium"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {}
	output_sockets = {
		"medium": sockets.MaxwellMediumSocketDef(
			label="Medium"
		),
	}
	
	####################
	# - Properties
	####################
	material: bpy.props.EnumProperty(
		name="",
		description="",
		#icon="NODE_MATERIAL",
		items=[
			(
				mat_key,
				td.material_library[mat_key].name,
				", ".join([
					ref.journal
					for ref in td.material_library[mat_key].variants[
						td.material_library[mat_key].default
					].reference
				])
			)
			for mat_key in td.material_library
			if mat_key != "graphene"  ## For some reason, it's unique...
		],
		default="Au",
		update=(lambda self,context: self.update()),
	)
	
	####################
	# - UI
	####################
	def draw_props(self, context, layout):
		layout.prop(self, "material", text="")
	
	def draw_info(self, context, layout):
		layout.operator(ExperimentOperator00.bl_idname, text="Experiment")
		vac_speed_of_light = sc.constants.speed_of_light * spu.meter/spu.second
		
		mat = td.material_library[self.material]
		freq_range = [
			spu.convert_to(
				val * spu.hertz,
				spuex.terahertz,
			) / spuex.terahertz
			for val in mat.medium.frequency_range
		]
		nm_range = [
			spu.convert_to(
				vac_speed_of_light / (val * spu.hertz),
				spu.nanometer,
			) / spu.nanometer
			for val in mat.medium.frequency_range
		]
		
		layout.label(text=f"nm: [{nm_range[1].n(2)}, {nm_range[0].n(2)}]")
		layout.label(text=f"THz: [{freq_range[0].n(2)}, {freq_range[1].n(2)}]")
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("medium")
	def compute_medium(self: contracts.NodeTypeProtocol) -> td.AbstractMedium:
		return td.material_library[self.material].medium
	
	####################
	# - Experiment
	####################
	def invoke_matplotlib_and_update_image(self):
		import matplotlib.pyplot as plt
		mat = td.material_library[self.material]
		
		aspect_ratio = 1.0
		for area in bpy.context.screen.areas:
			if area.type == 'IMAGE_EDITOR':
				width = area.width
				height = area.height
				aspect_ratio = width / height
		
		# Generate a plot with matplotlib
		fig_width = 6
		fig_height = fig_width / aspect_ratio
		fig, ax = plt.subplots(figsize=(fig_width, fig_height))
		ax.set_aspect(aspect_ratio)
		mat.medium.plot(
			np.linspace(*mat.medium.frequency_range[:2], 50),
			ax=ax,
		)
		
		# Save the plot to a temporary file
		temp_plot_file = bpy.path.abspath('//temp_plot.png')
		fig.savefig(temp_plot_file, bbox_inches='tight')
		plt.close(fig)  # Close the figure to free up memory
		
		# Load or reload the image in Blender
		if "matplotlib_plot" in bpy.data.images:
			image = bpy.data.images["matplotlib_plot"]
			image.reload()
		else:
			image = bpy.data.images.load(temp_plot_file)
			image.name = "matplotlib_plot"
		
		# Write the plot to an image datablock in Blender
		for area in bpy.context.screen.areas:
			if area.type == 'IMAGE_EDITOR':
				for space in area.spaces:
					if space.type == 'IMAGE_EDITOR':
						space.image = image
						return True



####################
# - Blender Registration
####################
BL_REGISTER = [
	ExperimentOperator00,
	LibraryMediumNode,
]
BL_NODES = {
	contracts.NodeType.LibraryMedium: (
		contracts.NodeCategory.MAXWELLSIM_MEDIUMS
	)
}
