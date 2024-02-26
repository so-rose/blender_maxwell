import sympy as sp
import sympy.physics.units as spu
import scipy as sc

from ... import contracts
from ... import sockets
from .. import base

class CombineNode(base.MaxwellSimTreeNode):
	node_type = contracts.NodeType.Combine
	bl_label = "Combine"
	#bl_icon = ...
	
	####################
	# - Sockets
	####################
	input_sockets = {}
	input_socket_sets = {
		"real_3d_vector": {
			f"x_{i}": sockets.RealNumberSocketDef(
				label=f"x_{i}"
			)
			for i in range(3)
		},
		"point_3d": {
			axis: sockets.PhysicalLengthSocketDef(
				label=axis
			)
			for i, axis in zip(
				range(3),
				["x", "y", "z"]
			)
		},
		"size_3d": {
			axis_key: sockets.PhysicalLengthSocketDef(
				label=axis_label
			)
			for i, axis_key, axis_label in zip(
				range(3),
				["x_size", "y_size", "z_size"],
				["X Size", "Y Size", "Z Size"],
			)
		},
	}
	output_sockets = {}
	output_socket_sets = {
		"real_3d_vector": {
			"real_3d_vector": sockets.Real3DVectorSocketDef(
				label="Real 3D Vector",
			),
		},
		"point_3d": {
			"point_3d": sockets.PhysicalPoint3DSocketDef(
				label="3D Point",
			),
		},
		"size_3d": {
			"size_3d": sockets.PhysicalSize3DSocketDef(
				label="3D Size",
			),
		},
	}
	
	####################
	# - Output Socket Computation
	####################
	@base.computes_output_socket("real_3d_vector")
	def compute_real_3d_vector(self: contracts.NodeTypeProtocol) -> sp.Expr:
		x1, x2, x3 = [
			self.compute_input(f"x_{i}")
			for i in range(3)
		]
		
		return (x1, x2, x3)
	
	@base.computes_output_socket("point_3d")
	def compute_point_3d(self: contracts.NodeTypeProtocol) -> sp.Expr:
		x, y, z = [
			self.compute_input(axis)
			#spu.convert_to(
			#	self.compute_input(axis),
			#	spu.meter,
			#) / spu.meter
			for axis in ["x", "y", "z"]
		]
		
		return sp.Matrix([x, y, z])# * spu.meter
	
	@base.computes_output_socket("size_3d")
	def compute_size_3d(self: contracts.NodeTypeProtocol) -> sp.Expr:
		x_size, y_size, z_size = [
			self.compute_input(axis)
			#spu.convert_to(
			#	self.compute_input(axis),
			#	spu.meter,
			#) / spu.meter
			for axis in ["x_size", "y_size", "z_size"]
		]
		
		return sp.Matrix([x_size, y_size, z_size])# * spu.meter



####################
# - Blender Registration
####################
BL_REGISTER = [
	CombineNode,
]
BL_NODES = {
	contracts.NodeType.Combine: (
		contracts.NodeCategory.MAXWELLSIM_UTILITIES
	)
}
