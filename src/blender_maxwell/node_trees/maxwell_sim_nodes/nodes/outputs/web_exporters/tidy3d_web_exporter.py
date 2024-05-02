import typing as typ

import bpy
import tidy3d as td

from blender_maxwell.services import tdcloud
from blender_maxwell.utils import bl_cache, logger

from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)


####################
# - Web Uploader / Loader / Runner / Releaser
####################
class UploadSimulation(bpy.types.Operator):
	bl_idname = ct.OperatorType.NodeUploadSimulation
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
			and context.node.inputs['Sim'].is_linked
		)

	def execute(self, context):
		node = context.node
		node.upload_sim()
		return {'FINISHED'}


class RunSimulation(bpy.types.Operator):
	bl_idname = ct.OperatorType.NodeRunSimulation
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
	bl_idname = ct.OperatorType.NodeReloadTrackedTask
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
		if (cloud_task := tdcloud.TidyCloudTasks.task(node.tracked_task_id)) is None:
			msg = "Tried to reload tracked task, but it doesn't exist"
			raise RuntimeError(msg)

		cloud_task = tdcloud.TidyCloudTasks.update_task(cloud_task)
		return {'FINISHED'}


class EstCostTrackedTask(bpy.types.Operator):
	bl_idname = ct.OperatorType.NodeEstCostTrackedTask
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
			task_info := tdcloud.TidyCloudTasks.task_info(context.node.tracked_task_id)
		) is None:
			msg = "Tried to estimate cost of tracked task, but it doesn't exist"
			raise RuntimeError(msg)

		node.cache_est_cost = task_info.cost_est()
		return {'FINISHED'}


class ReleaseTrackedTask(bpy.types.Operator):
	bl_idname = ct.OperatorType.ReleaseTrackedTask
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

	input_sockets: typ.ClassVar = {
		'Sim': sockets.MaxwellFDTDSimSocketDef(),
		'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=False,
		),
	}
	output_sockets: typ.ClassVar = {
		'Sim Data': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=True,
		),
		'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=True,
		),
	}

	####################
	# - Properties
	####################
	lock_tree: bool = bl_cache.BLField(False, prop_ui=True)
	tracked_task_id: str = bl_cache.BLField('', prop_ui=True)

	####################
	# - Computed
	####################
	@bl_cache.cached_bl_property(persist=False)
	def sim(self) -> td.Simulation | None:
		sim = self._compute_input('Sim')
		has_sim = not ct.FlowSignal.check(sim)

		if has_sim:
			sim.validate_pre_upload(source_required=True)
			return sim
		return None

	@bl_cache.cached_bl_property(persist=False)
	def total_monitor_data(self) -> float:
		if self.sim is not None:
			return sum(self.sim.monitors_data_size.values())
		return 0.0

	@bl_cache.cached_bl_property(persist=False)
	def est_cost(self) -> float | None:
		if self.tracked_task_id != '':
			task_info = tdcloud.TidyCloudTasks.task_info(self.tracked_task_id)
			if task_info is not None:
				return task_info.cost_est()

		return None

	####################
	# - Methods
	####################
	def upload_sim(self):
		if self.sim is None:
			msg = 'Tried to upload simulation, but none is attached'
			raise ValueError(msg)

		if (new_task := self._compute_input('Cloud Task')) is None or isinstance(
			new_task,
			tdcloud.CloudTask,
		):
			msg = (
				'Tried to upload simulation to new task, but existing task was selected'
			)
			raise ValueError(msg)

		# Create Cloud Task
		cloud_task = tdcloud.TidyCloudTasks.mk_task(
			task_name=new_task[0],
			cloud_folder=new_task[1],
			sim=self.sim,
			verbose=True,
		)

		# Declare to Cloud Task that it Exists Now
		## This will change the UI to not allow free-text input.
		## If the socket is linked, this errors.
		self.inputs['Cloud Task'].on_new_task_created(cloud_task)

		# Track the Newly Uploaded Task ID
		self.tracked_task_id = cloud_task.task_id

	def run_tracked_task(self):
		if (cloud_task := tdcloud.TidyCloudTasks.task(self.tracked_task_id)) is None:
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
			ct.OperatorType.NodeUploadSimulation,
			text='Upload',
		)
		tree_lock_icon = 'LOCKED' if self.lock_tree else 'UNLOCKED'
		row.prop(
			self, self.blfields['lock_tree'], toggle=True, icon=tree_lock_icon, text=''
		)

		# Row: Run Sim Buttons
		row = layout.row(align=True)
		row.operator(
			ct.OperatorType.NodeRunSimulation,
			text='Run',
		)
		if self.tracked_task_id:
			tree_lock_icon = 'LOOP_BACK'
			row.operator(
				ct.OperatorType.ReleaseTrackedTask,
				icon='LOOP_BACK',
				text='',
			)

	def draw_info(self, context, layout):
		# Connection Info
		auth_icon = 'CHECKBOX_HLT' if tdcloud.IS_AUTHENTICATED else 'CHECKBOX_DEHLT'
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
		if self.inputs['Sim'].is_linked:
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
			col.label(text=f'{self.total_monitor_data / 1_000_000:.2f}MB')

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
				ct.OperatorType.NodeReloadTrackedTask,
				text='',
				icon='FILE_REFRESH',
			)
			row.operator(
				ct.OperatorType.NodeEstCostTrackedTask,
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
			cost_est = f'{self.est_cost:.2f}' if self.est_cost >= 0 else 'TBD'
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
	# - Events
	####################
	@events.on_value_changed(prop_name='lock_tree', props={'lock_tree'})
	def on_lock_tree_changed(self, props):
		if props['lock_tree']:
			self.trigger_event(ct.FlowEvent.EnableLock)
			self.locked = False
			for bl_socket in self.inputs:
				if bl_socket.name == 'Sim':
					continue
				bl_socket.locked = False

		else:
			self.trigger_event(ct.FlowEvent.DisableLock)

	@events.on_value_changed(prop_name='tracked_task_id', props={'tracked_task_id'})
	def on_tracked_task_id_changed(self, props):
		if props['tracked_task_id']:
			self.inputs['Cloud Task'].locked = True

		else:
			self.total_monitor_data = bl_cache.Signal.InvalidateCache
			self.est_cost = bl_cache.Signal.InvalidateCache
			self.inputs['Cloud Task'].on_prepare_new_task()
			self.inputs['Cloud Task'].locked = False

	####################
	# - Output Methods
	####################
	## TODO: Retrieve simulation data if/when the simulation is done
	@events.computes_output_socket(
		'Cloud Task',
		input_sockets={'Cloud Task'},
	)
	def compute_cloud_task(self, input_sockets: dict) -> tdcloud.CloudTask | None:
		if isinstance(cloud_task := input_sockets['Cloud Task'], tdcloud.CloudTask):
			return cloud_task

		return None


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
	ct.NodeType.Tidy3DWebExporter: (ct.NodeCategory.MAXWELLSIM_OUTPUTS_WEBEXPORTERS)
}
