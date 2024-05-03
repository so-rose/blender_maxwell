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
# - Operators
####################
class RecomputeSimInfo(bpy.types.Operator):
	bl_idname = ct.OperatorType.NodeRecomputeSimInfo
	bl_label = 'Recompute Tidy3D Sim Info'
	bl_description = 'Recompute info for any currently attached sim info'

	@classmethod
	def poll(cls, context):
		return (
			# Check Tidy3DWebExporter is Accessible
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			# Check Sim is Available (aka. uploadeable)
			and context.node.sim_info_available
			and context.node.sim_info_invalidated
		)

	def execute(self, context):
		node = context.node

		# Rehydrate the Cache
		node.total_monitor_data = bl_cache.Signal.InvalidateCache
		node.is_sim_uploadable = bl_cache.Signal.InvalidateCache

		# Remove the Invalidation Marker
		## -> This is OK, since we manually guaranteed that it's available.
		node.sim_info_invalidated = False
		return {'FINISHED'}


class UploadSimulation(bpy.types.Operator):
	bl_idname = ct.OperatorType.NodeUploadSimulation
	bl_label = 'Upload Tidy3D Simulation'
	bl_description = 'Upload the attached (locked) simulation, such that it is ready to run on the Tidy3D cloud'

	@classmethod
	def poll(cls, context):
		return (
			# Check Tidy3D Cloud
			tdcloud.IS_AUTHENTICATED
			# Check Tidy3DWebExporter is Accessible
			and hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			# Check Sim is Available (aka. uploadeable)
			and context.node.is_sim_uploadable
			and context.node.uploaded_task_id == ''
		)

	def execute(self, context):
		node = context.node
		cloud_task = tdcloud.TidyCloudTasks.mk_task(
			task_name=node.new_cloud_task.task_name,
			cloud_folder=node.new_cloud_task.cloud_folder,
			sim=node.sim,
			verbose=True,
		)
		node.uploaded_task_id = cloud_task.task_id
		return {'FINISHED'}


class ReleaseUploadedTask(bpy.types.Operator):
	bl_idname = ct.OperatorType.NodeReleaseUploadedTask
	bl_label = 'Release Tracked Tidy3D Cloud Task'
	bl_description = 'Release the currently tracked simulation task'

	@classmethod
	def poll(cls, context):
		return (
			# Check Tidy3DWebExporter is Accessible
			hasattr(context, 'node')
			and hasattr(context.node, 'node_type')
			and context.node.node_type == ct.NodeType.Tidy3DWebExporter
			# Check Sim is Available (aka. uploadeable)
			and context.node.uploaded_task_id != ''
		)

	def execute(self, context):
		node = context.node
		node.uploaded_task_id = ''
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
		'Cloud Task': sockets.Tidy3DCloudTaskSocketDef(
			should_exist=True,
		),
	}

	####################
	# - Properties
	####################
	sim_info_available: bool = bl_cache.BLField(False)
	sim_info_invalidated: bool = bl_cache.BLField(False)
	uploaded_task_id: str = bl_cache.BLField('')

	####################
	# - Computed - Sim
	####################
	@bl_cache.cached_bl_property(persist=False)
	def sim(self) -> td.Simulation | None:
		sim = self._compute_input('Sim')
		has_sim = not ct.FlowSignal.check(sim)

		if has_sim:
			return sim
		return None

	@bl_cache.cached_bl_property(persist=False)
	def total_monitor_data(self) -> float | None:
		if self.sim is not None:
			return sum(self.sim.monitors_data_size.values())
		return None

	####################
	# - Computed - New Cloud Task
	####################
	@property
	def new_cloud_task(self) -> ct.NewSimCloudTask | None:
		"""Retrieve the current new cloud task from the input socket.

		If one can't be loaded, return None.
		"""
		new_cloud_task = self._compute_input(
			'Cloud Task',
			kind=ct.FlowKind.Value,
		)
		has_new_cloud_task = not ct.FlowSignal.check(new_cloud_task)

		if has_new_cloud_task and new_cloud_task.task_name != '':
			return new_cloud_task
		return None

	####################
	# - Computed - Uploaded Cloud Task
	####################
	@property
	def uploaded_task(self) -> tdcloud.CloudTask | None:
		"""Retrieve the uploaded cloud task.

		If one can't be loaded, return None.
		"""
		has_uploaded_task = self.uploaded_task_id != ''

		if has_uploaded_task:
			return tdcloud.TidyCloudTasks.task(self.uploaded_task_id)
		return None

	@property
	def uploaded_task_info(self) -> tdcloud.CloudTask | None:
		"""Retrieve the uploaded cloud task.

		If one can't be loaded, return None.
		"""
		has_uploaded_task = self.uploaded_task_id != ''

		if has_uploaded_task:
			return tdcloud.TidyCloudTasks.task_info(self.uploaded_task_id)
		return None

	@bl_cache.cached_bl_property(persist=False)
	def uploaded_est_cost(self) -> float | None:
		task_info = self.uploaded_task_info
		if task_info is not None:
			est_cost = task_info.cost_est()
			if est_cost is not None:
				return est_cost

		return None

	####################
	# - Computed - Combined
	####################
	@bl_cache.cached_bl_property(persist=False)
	def is_sim_uploadable(self) -> bool:
		if (
			self.sim is not None
			and self.uploaded_task_id == ''
			and self.new_cloud_task is not None
			and self.new_cloud_task.task_name != ''
		):
			try:
				self.sim.validate_pre_upload(source_required=True)
			except:
				log.exception()
				return False
			else:
				return True
		return False

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
		if self.uploaded_task_id:
			row.operator(
				ct.OperatorType.NodeReleaseUploadedTask,
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
		if self.sim is not None:
			row = layout.row()
			row.alignment = 'CENTER'
			row.label(text='Sim Info')
			box = layout.box()

			if self.sim_info_invalidated:
				box.operator(ct.OperatorType.NodeRecomputeSimInfo, text='Regenerate')
			else:
				split = box.split(factor=0.5)

				## Split: Left Column
				col = split.column(align=False)
				col.label(text='𝝨 Data')

				## Split: Right Column
				col = split.column(align=False)
				col.alignment = 'RIGHT'
				col.label(text=f'{self.total_monitor_data / 1_000_000:.2f}MB')

		if self.uploaded_task_info is not None:
			# Uploaded Task Information
			box = layout.box()
			split = box.split(factor=0.6)

			## Split: Left Column
			col = split.column(align=False)
			col.label(text='Status')
			col.label(text='Est. Cost')
			col.label(text='Real Cost')

			## Split: Right Column
			cost_est = (
				f'{self.uploaded_est_cost:.2f}'
				if self.uploaded_est_cost is not None
				else 'TBD'
			)
			cost_real = (
				f'{self.uploaded_task_info.cost_real:.2f}'
				if self.uploaded_task_info.cost_real is not None
				else 'TBD'
			)

			col = split.column(align=False)
			col.alignment = 'RIGHT'
			col.label(text=self.uploaded_task_info.status.capitalize())
			col.label(text=f'{cost_est} creds')
			col.label(text=f'{cost_real} creds')

		# Connection Information

	####################
	# - Events
	####################
	@events.on_value_changed(
		socket_name='Sim',
		run_on_init=True,
		props={'sim_info_available', 'sim_info_invalidated'},
	)
	def on_sim_changed(self, props) -> None:
		# Sim Linked | First Value Change
		if self.inputs['Sim'].is_linked and not props['sim_info_available']:
			log.critical('First Change: Mark Sim Info Available')
			self.sim = bl_cache.Signal.InvalidateCache
			self.total_monitor_data = bl_cache.Signal.InvalidateCache
			self.is_sim_uploadable = bl_cache.Signal.InvalidateCache
			self.sim_info_available = True

		# Sim Linked | Second Value Change
		if (
			self.inputs['Sim'].is_linked
			and props['sim_info_available']
			and not props['sim_info_invalidated']
		):
			log.critical('Second Change: Mark Sim Info Invalided')
			self.sim_info_invalidated = True

		# Sim Linked | Nth Time
		## -> Danger of infinite expensive recompute of the sim every change.
		## -> Instead, user must manually set "available & not invalidated".
		## -> The UI should explain that the caches are dry.
		## -> The UI should also provide such a "hydration" button.

		# Sim Not Linked
		## -> If the sim is straight-up not available, cache needs changing.
		## -> Luckily, since we know there's no sim, invalidation is cheap.
		## -> Ends up being a "circuit breaker" for sim_info_invalidated.
		elif not self.inputs['Sim'].is_linked:
			log.critical('Unlinked: Short Circuit Zap Cache')
			self.sim = bl_cache.Signal.InvalidateCache
			self.total_monitor_data = bl_cache.Signal.InvalidateCache
			self.is_sim_uploadable = bl_cache.Signal.InvalidateCache
			self.sim_info_available = False
			self.sim_info_invalidated = False

	@events.on_value_changed(
		socket_name='Cloud Task',
		run_on_init=True,
	)
	def on_new_cloud_task_changed(self):
		self.is_sim_uploadable = bl_cache.Signal.InvalidateCache

	@events.on_value_changed(
		# Trigger
		prop_name='uploaded_task_id',
		run_on_init=True,
		# Loaded
		props={'uploaded_task_id'},
	)
	def on_uploaded_task_changed(self, props):
		log.critical('Uploaded Task Changed')
		self.is_sim_uploadable = bl_cache.Signal.InvalidateCache

		if props['uploaded_task_id'] != '':
			self.trigger_event(ct.FlowEvent.EnableLock)
			self.locked = False

		else:
			self.trigger_event(ct.FlowEvent.DisableLock)

		max_tries = 10
		for _ in range(max_tries):
			self.uploaded_est_cost = bl_cache.Signal.InvalidateCache
			if self.uploaded_est_cost is not None:
				break

	####################
	# - Outputs
	####################
	@events.computes_output_socket(
		'Cloud Task',
		props={'uploaded_task_id', 'uploaded_task'},
	)
	def compute_cloud_task(self, props) -> tdcloud.CloudTask | None:
		if props['uploaded_task_id'] != '':
			return props['uploaded_task']

		return ct.FlowSignal.FlowPending


####################
# - Blender Registration
####################
BL_REGISTER = [
	UploadSimulation,
	ReleaseUploadedTask,
	Tidy3DWebExporterNode,
	RecomputeSimInfo,
]
BL_NODES = {
	ct.NodeType.Tidy3DWebExporter: (ct.NodeCategory.MAXWELLSIM_OUTPUTS_WEBEXPORTERS)
}
