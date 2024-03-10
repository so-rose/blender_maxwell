import json
import tempfile
import functools
import typing as typ
import json
from pathlib import Path

import bpy
import sympy as sp
import pydantic as pyd
import tidy3d as td
import tidy3d.web as _td_web

from ......utils.auth_td_web import g_td_web, is_td_web_authed
from .... import contracts as ct
from .... import sockets
from ... import base

####################
# - Task Getters
####################
## TODO: We should probably refactor this setup.
@functools.cache
def estimated_task_cost(task_id: str):
	return _td_web.api.webapi.estimate_cost(task_id)

@functools.cache
def billed_task_cost(task_id: str):
	return _td_web.api.webapi.real_cost(task_id)

@functools.cache
def task_status(task_id: str):
	task = _td_web.api.webapi.get_info(task_id)
	return task.status

####################
# - Progress Timer
####################
## TODO: We should probably refactor this too.
class Tidy3DTaskStatusModalOperator(bpy.types.Operator):
	bl_idname = "blender_maxwell.tidy_3d_task_status_modal_operator"
	bl_label = "Tidy3D Task Status Modal Operator"

	_timer = None
	_task_id = None
	_node = None
	_status = None
	_reported_done = False

	def modal(self, context, event):
		# Retrieve New Status
		task_status.cache_clear()
		new_status = task_status(self._task_id)
		if new_status != self._status:
			task_status.cache_clear()
			self._status = new_status
			
		# Check Done Status
		if self._status in {"success", "error"}:
			# Report Done
			if not self._reported_done:
				self._node.trigger_action("value_changed")
				self._reported_done = True
			
			# Finish when Billing is Known
			if not billed_task_cost(self._task_id):
				billed_task_cost.cache_clear()
			else:
				return {'FINISHED'}
		
		return {'PASS_THROUGH'}

	def execute(self, context):
		node = context.node
		wm = context.window_manager
		
		self._timer = wm.event_timer_add(0.25, window=context.window)
		self._task_id = node.uploaded_task_id
		self._node = node
		self._status = task_status(self._task_id)
		
		wm.modal_handler_add(self)
		return {'RUNNING_MODAL'}

####################
# - Web Uploader / Loader / Runner / Releaser
####################
## TODO: We should probably refactor this too.
class Tidy3DWebUploadOperator(bpy.types.Operator):
	bl_idname = "blender_maxwell.tidy_3d_web_upload_operator"
	bl_label = "Tidy3D Web Upload Operator"
	bl_description = "Upload the attached (locked) simulation, such that it is ready to run on the Tidy3D cloud"

	@classmethod
	def poll(cls, context):
		space = context.space_data
		return (
			space.type == 'NODE_EDITOR'
			and space.node_tree is not None
			and space.node_tree.bl_idname == "MaxwellSimTreeType"
			and is_td_web_authed()
			and hasattr(context, "node")
			and context.node.lock_tree
		)

	def execute(self, context):
		node = context.node
		node.web_upload()
		return {'FINISHED'}

class Tidy3DLoadUploadedOperator(bpy.types.Operator):
	bl_idname = "blender_maxwell.tidy_3d_load_uploaded_operator"
	bl_label = "Tidy3D Load Uploaded Operator"
	bl_description = "Load an already-uploaded simulation, as selected in the dropdown of the 'Cloud Task' socket"

	@classmethod
	def poll(cls, context):
		space = context.space_data
		return (
			space.type == 'NODE_EDITOR'
			and space.node_tree is not None
			and space.node_tree.bl_idname == "MaxwellSimTreeType"
			and is_td_web_authed()
			and hasattr(context, "node")
			and context.node.lock_tree
		)

	def execute(self, context):
		node = context.node
		node.load_uploaded_task()
		
		# Load Simulation to Compare
		## Load Local Sim
		local_sim = node._compute_input("FDTD Sim")
		
		## Load Cloud Sim
		task_id = node.compute_output("Cloud Task")
		with tempfile.NamedTemporaryFile(delete=False) as f:
			_path_tmp = Path(f.name)
			_path_tmp.rename(f.name + ".json")
			path_tmp = Path(f.name + ".json")
		cloud_sim = _td_web.api.webapi.load_simulation(task_id, path=str(path_tmp))
		
		Path(path_tmp).unlink()
		
		## Compare
		if local_sim != cloud_sim:
			node.release_uploaded_task()
			msg = "Loaded simulation doesn't match input simulation"
			raise ValueError(msg)
		
		return {'FINISHED'}

class RunUploadedTidy3DSim(bpy.types.Operator):
	bl_idname = "blender_maxwell.run_uploaded_tidy_3d_sim"
	bl_label = "Run Uploaded Tidy3D Sim"
	bl_description = "Run the currently uploaded (and loaded) simulation"

	@classmethod
	def poll(cls, context):
		space = context.space_data
		return (
			space.type == 'NODE_EDITOR'
			and space.node_tree is not None
			and space.node_tree.bl_idname == "MaxwellSimTreeType"
			and is_td_web_authed()
			and hasattr(context, "node")
			and context.node.lock_tree
			and context.node.uploaded_task_id
			and task_status(context.node.uploaded_task_id) == "draft"
		)

	def execute(self, context):
		node = context.node
		node.run_uploaded_task()
		bpy.ops.blender_maxwell.tidy_3d_task_status_modal_operator()
		return {'FINISHED'}

class ReleaseTidy3DExportOperator(bpy.types.Operator):
	bl_idname = "blender_maxwell.release_tidy_3d_export_operator"
	bl_label = "Release Tidy3D Export Operator"

	@classmethod
	def poll(cls, context):
		space = context.space_data
		return (
			space.type == 'NODE_EDITOR'
			and space.node_tree is not None
			and space.node_tree.bl_idname == "MaxwellSimTreeType"
			and is_td_web_authed()
			and hasattr(context, "node")
			and context.node.lock_tree
			and context.node.uploaded_task_id
		)

	def execute(self, context):
		node = context.node
		node.release_uploaded_task()
		return {'FINISHED'}



####################
# - Web Exporter Node
####################
class Tidy3DWebExporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DWebExporter
	bl_label = "Tidy3DWebExporter"
	
	input_sockets = {
		"FDTD Sim": sockets.MaxwellFDTDSimSocketDef(),
		"Cloud Task": sockets.Tidy3DCloudTaskSocketDef(
			task_exists=False,
		),
	}
	output_sockets = {
		"Cloud Task": sockets.Tidy3DCloudTaskSocketDef(
			task_exists=True,
		),
	}
	
	lock_tree: bpy.props.BoolProperty(
		name="Whether to lock the attached tree",
		description="Whether or not to lock the attached tree",
		default=False,
		update=(lambda self, context: self.sync_lock_tree(context)),
	)
	uploaded_task_id: bpy.props.StringProperty(
		name="Uploaded Task ID",
		description="The uploaded task ID",
		default="",
	)
	
	####################
	# - Sync Methods
	####################
	def sync_lock_tree(self, context):
		node_tree = self.id_data
		
		if self.lock_tree:
			self.trigger_action("enable_lock")
			self.locked = False
			for bl_socket in self.inputs:
				if bl_socket.name == "FDTD Sim": continue
				bl_socket.locked = False
		
		else:
			self.trigger_action("disable_lock")
	
	####################
	# - Output Socket Callbacks
	####################
	def web_upload(self):
		if not (sim := self._compute_input("FDTD Sim")):
			raise ValueError("Must attach simulation")
		
		if not (new_task_dict := self._compute_input("Cloud Task")):
			raise ValueError("No valid cloud task defined")
		
		td_web = g_td_web(None)  ## Presume already auth'ed
		
		self.uploaded_task_id = td_web.api.webapi.upload(
			sim,
			**new_task_dict,
			verbose=True,
		)
		
		self.inputs["Cloud Task"].sync_task_loaded(self.uploaded_task_id)
	
	def load_uploaded_task(self):
		self.inputs["Cloud Task"].sync_task_loaded(None)
		self.uploaded_task_id = self._compute_input("Cloud Task")
		
		self.trigger_action("value_changed")
	
	def run_uploaded_task(self):
		td_web = g_td_web(None)  ## Presume already auth'ed
		td_web.api.webapi.start(self.uploaded_task_id)
		
		self.trigger_action("value_changed")
	
	def release_uploaded_task(self):
		self.uploaded_task_id = ""
		self.inputs["Cloud Task"].sync_task_released(specify_new_task=True)
		
		self.trigger_action("value_changed")
	
	####################
	# - UI
	####################
	def draw_operators(self, context, layout):
		is_authed = is_td_web_authed()
		has_uploaded_task_id = bool(self.uploaded_task_id)
		
		# Row: Run Simulation
		row = layout.row(align=True)
		if has_uploaded_task_id: row.enabled = False
		row.operator(
			Tidy3DWebUploadOperator.bl_idname,
			text="Upload Sim",
		)
		tree_lock_icon = "LOCKED" if self.lock_tree else "UNLOCKED"
		row.prop(self, "lock_tree", toggle=True, icon=tree_lock_icon, text="")
		
		# Row: Run Simulation
		row = layout.row(align=True)
		if is_authed and has_uploaded_task_id:
			run_sim_text = f"Run Sim (~{estimated_task_cost(self.uploaded_task_id):.3f} credits)"
		else:
			run_sim_text = f"Run Sim"
		
		row.operator(
			RunUploadedTidy3DSim.bl_idname,
			text=run_sim_text,
		)
		if has_uploaded_task_id:
			tree_lock_icon = "LOOP_BACK"
			row.operator(
				ReleaseTidy3DExportOperator.bl_idname,
				icon="LOOP_BACK",
				text="",
			)
		else:
			row.operator(
				Tidy3DLoadUploadedOperator.bl_idname,
				icon="TRIA_UP_BAR",
				text="",
			)
		
		# Row: Simulation Progress
		if is_authed and has_uploaded_task_id:
			progress = {
				"draft": (0.0, "Waiting to Run..."),
				"initialized": (0.0, "Initializing..."),
				"queued": (0.0, "Queued..."),
				"preprocessing": (0.05, "Pre-processing..."),
				"running": (0.2, "Running..."),
				"postprocessing": (0.85, "Post-processing..."),
				"success": (1.0, f"Success (={billed_task_cost(self.uploaded_task_id)} credits)"),
				"error": (1.0, f"Error (={billed_task_cost(self.uploaded_task_id)} credits)"),
			}[task_status(self.uploaded_task_id)]
			
			layout.separator()
			row = layout.row(align=True)
			row.progress(
				factor=progress[0],
				type="BAR",
				text=progress[1],
			)
	
	####################
	# - Output Methods
	####################
	@base.computes_output_socket(
		"Cloud Task",
		input_sockets={"Cloud Task"},
	)
	def compute_cloud_task(self, input_sockets: dict) -> str | None:
		if self.uploaded_task_id: return self.uploaded_task_id
		return None
	
	####################
	# - Update
	####################
	@base.on_value_changed(socket_name="FDTD Sim")
	def on_value_changed__fdtd_sim(self):
		estimated_task_cost.cache_clear()
		task_status.cache_clear()
		billed_task_cost.cache_clear()
	
	@base.on_value_changed(socket_name="Cloud Task")
	def on_value_changed__cloud_task(self):
		estimated_task_cost.cache_clear()
		task_status.cache_clear()
		billed_task_cost.cache_clear()


####################
# - Blender Registration
####################
BL_REGISTER = [
	Tidy3DWebUploadOperator,
	Tidy3DTaskStatusModalOperator,
	RunUploadedTidy3DSim,
	Tidy3DLoadUploadedOperator,
	ReleaseTidy3DExportOperator,
	Tidy3DWebExporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebExporter: (
		ct.NodeCategory.MAXWELLSIM_OUTPUTS_EXPORTERS
	)
}
