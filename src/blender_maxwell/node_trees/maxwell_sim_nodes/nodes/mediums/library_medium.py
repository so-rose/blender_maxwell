import typing as typ

import bpy
import scipy as sc
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td

from .....utils import extra_sympy_units as spuex
from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base

VAC_SPEED_OF_LIGHT = sc.constants.speed_of_light * spu.meter / spu.second


class LibraryMediumNode(base.MaxwellSimNode):
	node_type = ct.NodeType.LibraryMedium
	bl_label = 'Library Medium'

	####################
	# - Sockets
	####################
	input_sockets = {}
	output_sockets = {
		'Medium': sockets.MaxwellMediumSocketDef(),
	}

	managed_obj_defs = {
		'nk_plot': ct.schemas.ManagedObjDef(
			mk=lambda name: managed_objs.ManagedBLImage(name),
			name_prefix='',
		)
	}

	####################
	# - Properties
	####################
	material: bpy.props.EnumProperty(
		name='',
		description='',
		# icon="NODE_MATERIAL",
		items=[
			(
				mat_key,
				td.material_library[mat_key].name,
				', '.join(
					[
						ref.journal
						for ref in td.material_library[mat_key]
						.variants[td.material_library[mat_key].default]
						.reference
					]
				),
			)
			for mat_key in td.material_library
			if mat_key != 'graphene'  ## For some reason, it's unique...
		],
		default='Au',
		update=(lambda self, context: self.sync_prop('material', context)),
	)

	@property
	def freq_range_str(self) -> tuple[sp.Expr, sp.Expr]:
		## TODO: Cache (node instances don't seem able to keep data outside of properties, not even cached_property)
		mat = td.material_library[self.material]
		freq_range = [
			spu.convert_to(
				val * spu.hertz,
				spuex.terahertz,
			)
			/ spuex.terahertz
			for val in mat.medium.frequency_range
		]
		return sp.pretty(
			[freq_range[0].n(4), freq_range[1].n(4)], use_unicode=True
		)

	@property
	def nm_range_str(self) -> str:
		## TODO: Cache (node instances don't seem able to keep data outside of properties, not even cached_property)
		mat = td.material_library[self.material]
		nm_range = [
			spu.convert_to(
				VAC_SPEED_OF_LIGHT / (val * spu.hertz),
				spu.nanometer,
			)
			/ spu.nanometer
			for val in reversed(mat.medium.frequency_range)
		]
		return sp.pretty(
			[nm_range[0].n(4), nm_range[1].n(4)], use_unicode=True
		)

	####################
	# - UI
	####################
	def draw_props(self, context, layout):
		layout.prop(self, 'material', text='')

	def draw_info(self, context, col):
		# UI Drawing
		split = col.split(factor=0.23, align=True)

		_col = split.column(align=True)
		_col.alignment = 'LEFT'
		_col.label(text='nm')
		_col.label(text='THz')

		_col = split.column(align=True)
		_col.alignment = 'RIGHT'
		_col.label(text=self.nm_range_str)
		_col.label(text=self.freq_range_str)

	####################
	# - Output Sockets
	####################
	@base.computes_output_socket('Medium')
	def compute_vac_wl(self) -> sp.Expr:
		return td.material_library[self.material].medium

	####################
	# - Event Callbacks
	####################
	@base.on_show_plot(
		managed_objs={'nk_plot'},
		props={'material'},
		stop_propagation=True,  ## Plot only the first plottable node
	)
	def on_show_plot(
		self,
		managed_objs: dict[str, ct.schemas.ManagedObj],
		props: dict[str, typ.Any],
	):
		medium = td.material_library[props['material']].medium
		freq_range = [
			spu.convert_to(
				val * spu.hertz,
				spuex.terahertz,
			)
			/ spu.hertz
			for val in medium.frequency_range
		]

		managed_objs['nk_plot'].mpl_plot_to_image(
			lambda ax: medium.plot(medium.frequency_range, ax=ax),
			bl_select=True,
		)


####################
# - Blender Registration
####################
BL_REGISTER = [
	LibraryMediumNode,
]
BL_NODES = {ct.NodeType.LibraryMedium: (ct.NodeCategory.MAXWELLSIM_MEDIUMS)}
