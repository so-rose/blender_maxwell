import typing as typ
from pathlib import Path

import bpy
import tidy3d as td

from ......utils import logger
from .... import contracts as ct
from .... import sockets
from ... import base, events

log = logger.get(__name__)

TD_FILE_EXTS = {
	'.hdf5.gz',
	'.hdf5',
	'.json',
	'.yaml',
}


####################
# - Node
####################
class Tidy3DFileImporterNode(base.MaxwellSimNode):
	node_type = ct.NodeType.Tidy3DFileImporter
	bl_label = 'Tidy3D File Importer'

	input_sockets: typ.ClassVar = {
		'File Path': sockets.FilePathSocketDef(),
	}

	tidy3d_type: bpy.props.EnumProperty(
		name='Tidy3D Type',
		description='Type of Tidy3D object to load',
		items=[
			(
				'SIMULATION_DATA',
				'Simulation Data',
				'Data from Completed Tidy3D Simulation',
			),
			('SIMULATION', 'Simulation', 'Tidy3D Simulation'),
			('MEDIUM', 'Medium', 'A Tidy3D Medium'),
		],
		default='SIMULATION_DATA',
		update=lambda self, context: self.sync_prop('tidy3d_type', context),
	)

	####################
	# - Event Methods: Compute Output Data
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
	# - Event Methods: Setup Output Socket
	####################
	@events.on_value_changed(
		socket_name='File Path',
		prop_name='tidy3d_type',
		input_sockets={'File Path'},
		props={'tidy3d_type'},
	)
	def on_file_changed(self, input_sockets: dict, props: dict):
		file_ext = ''.join(input_sockets['File Path'].suffixes)
		if not (input_sockets['File Path'].is_file() and file_ext in TD_FILE_EXTS):
			self.loose_output_sockets = {}
		else:
			self.loose_output_sockets = {
				'SIMULATION_DATA': {
					'Sim Data': sockets.MaxwellFDTDSimSocketDef(),
				},
				'SIMULATION': {'Sim': sockets.MaxwellFDTDSimDataSocketDef()},
				'MEDIUM': {'Medium': sockets.MaxwellMediumSocketDef()},
			}[props['tidy3d_type']]


####################
# - Blender Registration
####################
BL_REGISTER = [
	Tidy3DFileImporterNode,
]
BL_NODES = {
	ct.NodeType.Tidy3DFileImporter: (ct.NodeCategory.MAXWELLSIM_INPUTS_IMPORTERS)
}
