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

from ......utils import tdcloud
from .... import contracts as ct
from .... import sockets
from ... import base


####################
# - Web Uploader / Loader / Runner / Releaser
####################
class UploadSimulation(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodes__upload_simulation'
	bl_label = 'Upload Tidy3D Simulation'
	bl_description = 'Upload the attached (locked) simulation, such that it is ready to run on the Tidy3D cloud'

	@classmethod
	def poll(cls, context):
		return (
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			and context.node.lock_tree
			and tdcloud.IS_AUTHENTICATED
			and not context.node.tracked_task_id
			and context.node.inputs['FDTD Sim'].is_linked
		)

	def execute(self, context):
		node = context.node
		node.upload_sim()
		return {'FINISHED'}


class RunSimulation(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodes__run_simulation'
	bl_label = 'Run Tracked Tidy3D Sim'
	bl_description = 'Run the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		return (
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			and tdcloud.IS_AUTHENTICATED
			and context.node.tracked_task_id
			and (
				task_info := tdcloud.TidyCloudTasks.task_info(
					context.node.tracked_task_id
				)
			)
			is not None
			and task_info.status == 'draft'
		)

	def execute(self, context):
		node = context.node
		node.run_tracked_task()
		return {'FINISHED'}


class ReloadTrackedTask(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodes__reload_tracked_task'
	bl_label = 'Reload Tracked Tidy3D Cloud Task'
	bl_description = 'Reload the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		return (
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			and tdcloud.IS_AUTHENTICATED
			and context.node.tracked_task_id
		)

	def execute(self, context):
		node = context.node
		if (
			cloud_task := tdcloud.TidyCloudTasks.task(node.tracked_task_id)
		) is None:
			msg = "Tried to reload tracked task, but it doesn't exist"
			raise RuntimeError(msg)

		cloud_task = tdcloud.TidyCloudTasks.update_task(cloud_task)
		return {'FINISHED'}


class EstCostTrackedTask(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodes__est_cost_tracked_task'
	bl_label = 'Est Cost of Tracked Tidy3D Cloud Task'
	bl_description = 'Reload the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		return (
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			and tdcloud.IS_AUTHENTICATED
			and context.node.tracked_task_id
		)

	def execute(self, context):
		node = context.node
		if (
			task_info := tdcloud.TidyCloudTasks.task_info(
				context.node.tracked_task_id
			)
		) is None:
			msg = (
				"Tried to estimate cost of tracked task, but it doesn't exist"
			)
			raise RuntimeError(msg)

		node.cache_est_cost = task_info.cost_est()
		return {'FINISHED'}


class ReleaseTrackedTask(bpy.types.Operator):
	bl_idname = 'blender_maxwell.nodes__release_tracked_task'
	bl_label = 'Release Tracked Tidy3D Cloud Task'
	bl_description = 'Release the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		return (
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			# and tdcloud.IS_AUTHENTICATED
			and context.node.tracked_task_id
		)

	def execute(self, context):
		node = context.node
		node.tracked_task_id = ''
		return {'FINISHED'}


####################
# - Node
####################
class Tidy3DWebExporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DWebExporter
	bl_label = 'Tidy3D Web Exporter'

	input_sockets = {
		'FDTD Sim': sockets.MaxwellFDTDSimSocketDef(),
		'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=False,
		),
	}

	####################
	# - Properties
	####################
	lock_tree: bpy.props.BoolProperty(
		name='Whether to lock the attached tree',
		description='Whether or not to lock the attached tree',
		default=False,
		update=lambda self, context: self.sync_lock_tree(context),
	)
	tracked_task_id: bpy.props.StringProperty(
		name='Tracked Task ID',
		description='The currently tracked task ID',
		default='',
		update=lambda self, context: self.sync_tracked_task_id(context),
	)

	# Cache
	cache_total_monitor_data: bpy.props.FloatProperty(
		name='(Cached) Total Monitor Data',
		description='Required storage space by all monitors',
		default=0.0,
	)
	cache_est_cost: bpy.props.FloatProperty(
		name='(Cached) Estimated Total Cost',
		description='Est. Cost in FlexCompute units',
		default=-1.0,
	)

	####################
	# - Sync Methods
	####################
	def sync_lock_tree(self, context):
		if self.lock_tree:
			self.trigger_action('enable_lock')
			self.locked = False
			for bl_socket in self.inputs:
				if bl_socket.name == 'FDTD Sim':
					continue
				bl_socket.locked = False

		else:
			self.trigger_action('disable_lock')

		self.sync_prop('lock_tree', context)

	def sync_tracked_task_id(self, context):
		# Select Tracked Task
		if self.tracked_task_id:
			cloud_task = tdcloud.TidyCloudTasks.task(self.tracked_task_id)
			task_info = tdcloud.TidyCloudTasks.task_info(self.tracked_task_id)

			self.loose_output_sockets = {
				'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
					should_exist=True,
				),
			}
			self.inputs['Cloud Task'].locked = True

		# Release Tracked Task
		else:
			self.cache_est_cost = -1.0
			self.loose_output_sockets = {}
			self.inputs['Cloud Task'].sync_prepare_new_task()
			self.inputs['Cloud Task'].locked = False

		self.sync_prop('tracked_task_id', context)

	####################
	# - Output Socket Callbacks
	####################
	def validate_sim(self):
		if (sim := self._compute_input('FDTD Sim')) is None:
			msg = 'Tried to validate simulation, but none is attached'
			raise ValueError(msg)

		sim.validate_pre_upload(source_required=True)

	def upload_sim(self):
		if (sim := self._compute_input('FDTD Sim')) is None:
			msg = 'Tried to upload simulation, but none is attached'
			raise ValueError(msg)

		if (
			new_task := self._compute_input('Cloud Task')
		) is None or isinstance(
			new_task,
			tdcloud.CloudTask,
		):
			msg = 'Tried to upload simulation to new task, but existing task was selected'
			raise ValueError(msg)

		# Create Cloud Task
		cloud_task = tdcloud.TidyCloudTasks.mk_task(
			task_name=new_task[0],
			cloud_folder=new_task[1],
			sim=sim,
			upload_progress_cb=lambda uploaded_bytes: None,  ## TODO: Use!
			verbose=True,
		)

		# Declare to Cloud Task that it Exists Now
		## This will change the UI to not allow free-text input.
		## If the socket is linked, this errors.
		self.inputs['Cloud Task'].sync_created_new_task(cloud_task)

		# Track the Newly Uploaded Task ID
		self.tracked_task_id = cloud_task.task_id

	def run_tracked_task(self):
		if (
			cloud_task := tdcloud.TidyCloudTasks.task(self.tracked_task_id)
		) is None:
			msg = "Tried to run tracked task, but it doesn't exist"
			raise RuntimeError(msg)

		cloud_task.submit()
		tdcloud.TidyCloudTasks.update_task(
			cloud_task
		)  ## TODO: Check that status is actually immediately updated.

	####################
	# - UI
	####################
	def draw_operators(self, context, layout):
		# Row: Upload Sim Buttons
		row = layout.row(align=True)
		row.operator(
			UploadSimulation.bl_idname,
			text='Upload',
		)
		tree_lock_icon = 'LOCKED' if self.lock_tree else 'UNLOCKED'
		row.prop(self, 'lock_tree', toggle=True, icon=tree_lock_icon, text='')

		# Row: Run Sim Buttons
		row = layout.row(align=True)
		row.operator(
			RunSimulation.bl_idname,
			text='Run',
		)
		if self.tracked_task_id:
			tree_lock_icon = 'LOOP_BACK'
			row.operator(
				ReleaseTrackedTask.bl_idname,
				icon='LOOP_BACK',
				text='',
			)

	def draw_info(self, context, layout):
		# Connection Info
		auth_icon = (
			'CHECKBOX_HLT' if tdcloud.IS_AUTHENTICATED else 'CHECKBOX_DEHLT'
		)
		conn_icon = 'CHECKBOX_HLT' if tdcloud.IS_ONLINE else 'CHECKBOX_DEHLT'

		row = layout.row()
		row.alignment = 'CENTER'
		row.label(text='Cloud Status')
		box = layout.box()
		split = box.split(factor=0.85)

		## Split: Left Column
		col = split.column(align=False)
		col.label(text='Authed')
		col.label(text='Connected')

		## Split: Right Column
		col = split.column(align=False)
		col.label(icon=auth_icon)
		col.label(icon=conn_icon)

		# Simulation Info
		if self.inputs['FDTD Sim'].is_linked:
			row = layout.row()
			row.alignment = 'CENTER'
			row.label(text='Sim Info')
			box = layout.box()
			split = box.split(factor=0.4)

			## Split: Left Column
			col = split.column(align=False)
			col.label(text='𝝨 Output')

			## Split: Right Column
			col = split.column(align=False)
			col.alignment = 'RIGHT'
			col.label(
				text=f'{self.cache_total_monitor_data / 1_000_000:.2f}MB'
			)

		# Cloud Task Info
		if self.tracked_task_id and tdcloud.IS_AUTHENTICATED:
			task_info = tdcloud.TidyCloudTasks.task_info(self.tracked_task_id)
			if task_info is None:
				return

			## Header
			row = layout.row()
			row.alignment = 'CENTER'
			row.label(text='Task Info')

			## Progress Bar
			row = layout.row(align=True)
			row.progress(
				factor=0.0,
				type='BAR',
				text=f'Status: {task_info.status.capitalize()}',
			)
			row.operator(
				ReloadTrackedTask.bl_idname,
				text='',
				icon='FILE_REFRESH',
			)
			row.operator(
				EstCostTrackedTask.bl_idname,
				text='',
				icon='SORTTIME',
			)

			## Information
			box = layout.box()
			split = box.split(factor=0.4)

			## Split: Left Column
			col = split.column(align=False)
			col.label(text='Status')
			col.label(text='Est. Cost')
			col.label(text='Real Cost')

			## Split: Right Column
			cost_est = (
				f'{self.cache_est_cost:.2f}'
				if self.cache_est_cost >= 0
				else 'TBD'
			)
			cost_real = (
				f'{task_info.cost_real:.2f}'
				if task_info.cost_real is not None
				else 'TBD'
			)

			col = split.column(align=False)
			col.alignment = 'RIGHT'
			col.label(text=task_info.status.capitalize())
			col.label(text=f'{cost_est} creds')
			col.label(text=f'{cost_real} creds')

		# Connection Information

	####################
	# - Output Methods
	####################
	@base.computes_output_socket(
		'Cloud Task',
		input_sockets={'Cloud Task'},
	)
	def compute_cloud_task(
		self, input_sockets: dict
	) -> tdcloud.CloudTask | None:
		if isinstance(
			cloud_task := input_sockets['Cloud Task'], tdcloud.CloudTask
		):
			return cloud_task

		return None

	####################
	# - Output Methods
	####################
	@base.on_value_changed(
		socket_name='FDTD Sim',
		input_sockets={'FDTD Sim'},
	)
	def on_value_changed__fdtd_sim(self, input_sockets):
		if (sim := self._compute_input('FDTD Sim')) is None:
			self.cache_total_monitor_data = 0
			return

		sim.validate_pre_upload(source_required=True)
		self.cache_total_monitor_data = sum(sim.monitors_data_size.values())


####################
# - Blender Registration
####################
BL_REGISTER = [
	UploadSimulation,
	RunSimulation,
	ReloadTrackedTask,
	EstCostTrackedTask,
	ReleaseTrackedTask,
	Tidy3DWebExporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebExporter: (
		ct.NodeCategory.MAXWELLSIM_OUTPUTS_EXPORTERS
	)
}
