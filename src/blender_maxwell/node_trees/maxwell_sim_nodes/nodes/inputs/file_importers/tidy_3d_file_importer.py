import typing as typ
from pathlib import Path

import bpy
import tidy3d as td
import tidy3d.plugins.dispersion as td_dispersion

from blender_maxwell.utils import logger

from .... import contracts as ct
from .... import managed_objs, sockets
from ... import base, events

log = logger.get(__name__)

VALID_FILE_EXTS = {
	'SIMULATION_DATA': {
		'.hdf5.gz',
		'.hdf5',
	},
	'SIMULATION': {
		'.hdf5.gz',
		'.hdf5',
		'.json',
		'.yaml',
	},
	'MEDIUM': {
		'.hdf5.gz',
		'.hdf5',
		'.json',
		'.yaml',
	},
	'EXPERIM_DISP_MEDIUM': {
		'.txt',
	},
}

CACHE = {}


####################
# - Node
####################
class Tidy3DFileImporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DFileImporter
	bl_label = 'Tidy3D File Importer'

	input_sockets: typ.ClassVar = {
		'File Path': sockets.FilePathSocketDef(),
	}
	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	####################
	# - Properties
	####################
	## TODO: More automatic determination of which file type is in use :)
	tidy3d_type: bpy.props.EnumProperty(
		name='Tidy3D Type',
		description='Type of Tidy3D object to load',
		items=[
			(
				'SIMULATION_DATA',
				'Sim Data',
				'Data from Completed Tidy3D Simulation',
			),
			('SIMULATION', 'Sim', 'Tidy3D Simulation'),
			('MEDIUM', 'Medium', 'A Tidy3D Medium'),
			(
				'EXPERIM_DISP_MEDIUM',
				'Experim Disp Medium',
				'A pole-residue fit of experimental dispersive medium data, described by a .txt file specifying wl, n, k',
			),
		],
		default='SIMULATION_DATA',
		update=lambda self, context: self.sync_prop('tidy3d_type', context),
	)

	disp_fit__min_poles: bpy.props.IntProperty(
		name='min Poles',
		description='Min. # poles to fit to the experimental dispersive medium data',
		default=1,
	)
	disp_fit__max_poles: bpy.props.IntProperty(
		name='max Poles',
		description='Max. # poles to fit to the experimental dispersive medium data',
		default=5,
	)
	## TODO: Bool of whether to fit eps_inf, with conditional choice of eps_inf as socket
	disp_fit__tolerance_rms: bpy.props.FloatProperty(
		name='Max RMS',
		description='The RMS error threshold, below which the fit should be considered converged',
		default=0.001,
		precision=5,
	)
	## TODO: "AdvanceFastFitterParam" options incl. loss_bounds, weights, show_progress, show_unweighted_rms, relaxed, smooth, logspacing, numiters, passivity_num_iters, and slsqp_constraint_scale

	def draw_props(self, _: bpy.types.Context, col: bpy.types.UILayout):
		col.prop(self, 'tidy3d_type', text='')
		if self.tidy3d_type == 'EXPERIM_DISP_MEDIUM':
			row = col.row(align=True)
			row.alignment = 'CENTER'
			row.label(text='Pole-Residue Fit')

			col.prop(self, 'disp_fit__min_poles')
			col.prop(self, 'disp_fit__max_poles')
			col.prop(self, 'disp_fit__tolerance_rms')

	####################
	# - Event Methods: Output Data
	####################
	def _compute_sim_data_for(
		self, output_socket_name: str, file_path: Path
	) -> td.components.base.Tidy3dBaseModel:
		return {
			'Sim': td.Simulation,
			'Sim Data': td.SimulationData,
			'Medium': td.Medium,
		}[output_socket_name].from_file(str(file_path))

	@events.computes_output_socket(
		'Sim',
		input_sockets={'File Path'},
	)
	def compute_sim(self, input_sockets: dict) -> td.Simulation:
		return self._compute_sim_data_for('Sim', input_sockets['File Path'])

	@events.computes_output_socket(
		'Sim Data',
		input_sockets={'File Path'},
	)
	def compute_sim_data(self, input_sockets: dict) -> td.SimulationData:
		return self._compute_sim_data_for('Sim Data', input_sockets['File Path'])

	@events.computes_output_socket(
		'Medium',
		input_sockets={'File Path'},
	)
	def compute_medium(self, input_sockets: dict) -> td.Medium:
		return self._compute_sim_data_for('Medium', input_sockets['File Path'])

	####################
	# - Event Methods: Output Data | Dispersive Media
	####################
	@events.computes_output_socket(
		'Experim Disp Medium',
		input_sockets={'File Path'},
	)
	def compute_experim_disp_medium(self, input_sockets: dict) -> td.Medium:
		if CACHE.get(self.bl_label) is not None:
			log.debug('Reusing Cached Dispersive Medium')
			return CACHE[self.bl_label]['model']

		log.info('Loading Experimental Data')
		dispersion_fitter = td_dispersion.FastDispersionFitter.from_file(
			str(input_sockets['File Path'])
		)

		log.info('Computing Fast Dispersive Fit of Experimental Data...')
		pole_residue_medium, rms_error = dispersion_fitter.fit(
			min_num_poles=self.disp_fit__min_poles,
			max_num_poles=self.disp_fit__max_poles,
			tolerance_rms=self.disp_fit__tolerance_rms,
		)
		log.info('Fit Succeeded w/RMS "%s"!', f'{rms_error:.5f}')

		# Populate Cache
		CACHE[self.bl_label] = {}
		CACHE[self.bl_label]['model'] = pole_residue_medium
		CACHE[self.bl_label]['fitter'] = dispersion_fitter
		CACHE[self.bl_label]['rms_error'] = rms_error

		return pole_residue_medium

	####################
	# - Event Methods: Setup Output Socket
	####################
	@events.on_value_changed(
		socket_name='File Path',
		prop_name='tidy3d_type',
		input_sockets={'File Path'},
		props={'tidy3d_type'},
	)
	def on_file_changed(self, input_sockets: dict, props: dict):
		if CACHE.get(self.bl_label) is not None:
			del CACHE[self.bl_label]

		file_ext = ''.join(input_sockets['File Path'].suffixes)
		if not (
			input_sockets['File Path'].is_file()
			and file_ext in VALID_FILE_EXTS[props['tidy3d_type']]
		):
			self.loose_output_sockets = {}
		else:
			self.loose_output_sockets = {
				'SIMULATION_DATA': {
					'Sim Data': sockets.MaxwellFDTDSimDataSocketDef(),
				},
				'SIMULATION': {'Sim': sockets.MaxwellFDTDSimSocketDef()},
				'MEDIUM': {'Medium': sockets.MaxwellMediumSocketDef()},
				'EXPERIM_DISP_MEDIUM': {
					'Experim Disp Medium': sockets.MaxwellMediumSocketDef()
				},
			}[props['tidy3d_type']]

	####################
	# - Event Methods: Plot
	####################
	@events.on_show_plot(
		managed_objs={'plot'},
		props={'tidy3d_type'},
	)
	def on_show_plot(
		self,
		props: dict,
		managed_objs: dict,
	):
		"""When the filetype is 'Experimental Dispersive Medium', plot the computed model against the input data."""
		if props['tidy3d_type'] == 'EXPERIM_DISP_MEDIUM':
			# Populate Cache
			if CACHE.get(self.bl_label) is None:
				model_medium = self.compute_experim_disp_medium()
				disp_fitter = CACHE[self.bl_label]['fitter']
			else:
				model_medium = CACHE[self.bl_label]['model']
				disp_fitter = CACHE[self.bl_label]['fitter']

			# Plot
			managed_objs['plot'].mpl_plot_to_image(
				lambda ax: disp_fitter.plot(
					medium=model_medium,
					ax=ax,
				),
				bl_select=True,
			)


####################
# - Blender Registration
####################
BL_REGISTER = [
	Tidy3DFileImporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DFileImporter: (ct.NodeCategory.MAXWELLSIM_INPUTS_FILEIMPORTERS)
}
