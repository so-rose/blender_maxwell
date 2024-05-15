# blender_maxwell
# Copyright (C) 2024 blender_maxwell Project Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import enum
import typing as typ

import bpy
import sympy as sp
import sympy.physics.units as spu
import tidy3d as td
from tidy3d.material_library.material_library import MaterialItem as Tidy3DMediumItem
from tidy3d.material_library.material_library import VariantItem as Tidy3DMediumVariant

from blender_maxwell.utils import bl_cache, sci_constants
from blender_maxwell.utils import extra_sympy_units as spux

from ... import contracts as ct
from ... import managed_objs, sockets
from .. import base, events

_mat_lib_iter = iter(td.material_library)
_mat_key = ''


class VendoredMedium(enum.StrEnum):
	# Declare StrEnum of All Tidy3D Mediums
	## -> This is a 'for ... in ...', which uses globals as loop variables.
	## -> It's a bit of a hack, but very effective.
	while True:
		try:
			globals()['_mat_key'] = next(_mat_lib_iter)
		except StopIteration:
			break

		## -> Exclude graphene. Graphene is special.
		if _mat_key != 'graphene':
			locals()[_mat_key] = _mat_key

	@staticmethod
	def to_name(v: typ.Self) -> str:
		return td.material_library[v].name

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		return ''

	####################
	# - Medium Properties
	####################
	@property
	def tidy3d_medium_item(self) -> Tidy3DMediumItem:
		"""Extracts the Tidy3D "Medium Item", which encapsulates all the provided experimental variants."""
		return td.material_library[self]

	####################
	# - Medium Variant Properties
	####################
	@property
	def medium_variants(self) -> set[Tidy3DMediumVariant]:
		"""Extracts the list of medium variants, each corresponding to a particular experiment in the literature."""
		return self.tidy3d_medium_item.variants

	@property
	def default_medium_variant(self) -> Tidy3DMediumVariant:
		"""Extracts the "default" medium variant, as selected by Tidy3D."""
		return self.medium_variants[self.tidy3d_medium_item.default]

	####################
	# - Enum Helper
	####################
	@property
	def variants_as_bl_enum_elements(self) -> list[ct.BLEnumElement]:
		"""Computes a list of variants in a format suitable for use in a dynamic `EnumProperty`.

		Notes:
			This `EnumProperty` will only return a string `variant_name`.

			To reconstruct the actual `Tidy3DMediumVariant` object, one must therefore access it via the `vendored_medium.medium_variants[variant_name]`.
		"""
		return [
			(
				variant_name,
				variant_name,
				' | '.join([ref.journal for ref in variant.reference]),
				'',
				i,
			)
			for i, (variant_name, variant) in enumerate(self.medium_variants.items())
		]


class LibraryMediumNode(base.MaxwellSimNode):
	node_type = ct.NodeType.LibraryMedium
	bl_label = 'Library Medium'

	####################
	# - Sockets
	####################
	input_sockets: typ.ClassVar = {}
	output_sockets: typ.ClassVar = {
		'Medium': sockets.MaxwellMediumSocketDef(),
		'Valid Freqs': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.LazyArrayRange,
			physical_type=spux.PhysicalType.Freq,
		),
		'Valid WLs': sockets.ExprSocketDef(
			active_kind=ct.FlowKind.LazyArrayRange,
			physical_type=spux.PhysicalType.Length,
		),
	}

	managed_obj_types: typ.ClassVar = {
		'plot': managed_objs.ManagedBLImage,
	}

	####################
	# - Properties
	####################
	vendored_medium: VendoredMedium = bl_cache.BLField(VendoredMedium.Au)
	variant_name: enum.StrEnum = bl_cache.BLField(
		enum_cb=lambda self, _: self.search_variants()
	)

	def search_variants(self) -> list[ct.BLEnumElement]:
		"""Search for all valid variant of the current `self.vendored_medium`."""
		return self.vendored_medium.variants_as_bl_enum_elements

	####################
	# - Computed
	####################
	@bl_cache.cached_bl_property(depends_on={'vendored_medium', 'variant_name'})
	def variant(self) -> Tidy3DMediumVariant:
		"""Deduce the actual medium variant from `self.vendored_medium` and `self.variant_name`."""
		return self.vendored_medium.medium_variants[self.variant_name]

	@bl_cache.cached_bl_property(depends_on={'variant'})
	def medium(self) -> td.PoleResidue:
		"""Deduce the actual currently selected `PoleResidue` medium from `self.variant`."""
		return self.variant.medium

	@bl_cache.cached_bl_property(depends_on={'variant'})
	def data_url(self) -> str | None:
		"""Deduce the URL associated with the currently selected medium from `self.variant`."""
		return self.variant.data_url

	@bl_cache.cached_bl_property(depends_on={'variant'})
	def references(self) -> td.PoleResidue:
		"""Deduce the references associated with the currently selected `PoleResidue` medium from `self.variant`."""
		return self.variant.reference

	@bl_cache.cached_bl_property(depends_on={'medium'})
	def freq_range(self) -> sp.Expr:
		"""Deduce the frequency range as a unit-aware (THz, for convenience) column vector.

		A rational approximation to each frequency bound is computed with `sp.nsimplify`, in order to **guarantee** lack of precision-loss as computations are performed on the frequency.

		"""
		return spu.convert_to(
			sp.Matrix([sp.nsimplify(el) for el in self.medium.frequency_range])
			* spu.hertz,
			spux.terahertz,
		)

	@bl_cache.cached_bl_property(depends_on={'freq_range'})
	def wl_range(self) -> sp.Expr:
		"""Deduce the vacuum wavelength range as a unit-aware (nanometer, for convenience) column vector."""
		return sp.Matrix(
			self.freq_range.applyfunc(
				lambda el: spu.convert_to(
					sci_constants.vac_speed_of_light / el, spu.nanometer
				)
			)[::-1]
		)

	####################
	# - Cached UI Properties
	####################
	@staticmethod
	def _ui_range_format(sp_number: spux.SympyExpr, e_not_limit: int = 6):
		if sp_number.is_infinite:
			return sp.pretty(sp_number, use_unicode=True)

		number = float(sp_number.subs({spux.THz: 1, spu.nm: 1}))
		formatted_str = f'{number:.2f}'
		if len(formatted_str) > e_not_limit:
			formatted_str = f'{number:.2e}'
		return formatted_str

	@bl_cache.cached_bl_property(depends_on={'freq_range'})
	def ui_freq_range(self) -> tuple[str, str]:
		"""Cached mirror of `self.wl_range` which contains UI-ready strings."""
		return tuple([self._ui_range_format(el) for el in self.freq_range])

	@bl_cache.cached_bl_property(depends_on={'wl_range'})
	def ui_wl_range(self) -> tuple[str, str]:
		"""Cached mirror of `self.wl_range` which contains UI-ready strings."""
		return tuple([self._ui_range_format(el) for el in self.wl_range])

	####################
	# - UI
	####################
	def draw_label(self) -> str:
		return f'Medium: {self.vendored_medium}'

	def draw_props(self, _: bpy.types.Context, layout: bpy.types.UILayout) -> None:
		layout.prop(self, self.blfields['vendored_medium'], text='')
		layout.prop(self, self.blfields['variant_name'], text='')

	def draw_info(self, _: bpy.types.Context, col: bpy.types.UILayout) -> None:
		box = col.box()

		row = box.row(align=True)
		row.alignment = 'CENTER'
		row.label(text='min|max')

		grid = box.grid_flow(row_major=True, columns=2, align=True)
		grid.label(text='λ Range')
		grid.label(text='𝑓 Range')

		grid.label(text=self.ui_wl_range[0])
		grid.label(text=self.ui_freq_range[0])
		grid.label(text=self.ui_wl_range[1])
		grid.label(text=self.ui_freq_range[1])

		# URL Link
		if self.data_url is not None:
			box.operator('wm.url_open', text='Link to Data').url = self.data_url

	####################
	# - Events
	####################
	@events.on_value_changed(
		prop_name={'vendored_medium', 'variant_name'},
		run_on_init=True,
		props={'vendored_medium'},
	)
	def on_medium_changed(self, props):
		if self.variant_name not in props['vendored_medium'].medium_variants:
			self.variant_name = bl_cache.Signal.ResetEnumItems

		self.ui_freq_range = bl_cache.Signal.InvalidateCache
		self.ui_wl_range = bl_cache.Signal.InvalidateCache

	####################
	# - Output
	####################
	@events.computes_output_socket(
		'Medium',
		props={'medium'},
	)
	def compute_medium(self, props) -> sp.Expr:
		return props['medium']

	@events.computes_output_socket(
		'Valid Freqs',
		props={'freq_range'},
	)
	def compute_valid_freqs(self, props) -> sp.Expr:
		return props['freq_range']

	@events.computes_output_socket(
		'Valid Freqs',
		kind=ct.FlowKind.LazyArrayRange,
		props={'freq_range'},
	)
	def compute_valid_freqs_lazy(self, props) -> sp.Expr:
		return ct.LazyArrayRangeFlow(
			start=props['freq_range'][0] / spux.THz,
			stop=props['freq_range'][1] / spux.THz,
			steps=0,
			scaling=ct.ScalingMode.Lin,
			unit=spux.THz,
		)

	@events.computes_output_socket(
		'Valid WLs',
		props={'wl_range'},
	)
	def compute_valid_wls(self, props) -> sp.Expr:
		return props['wl_range']

	@events.computes_output_socket(
		'Valid WLs',
		kind=ct.FlowKind.LazyArrayRange,
		props={'wl_range'},
	)
	def compute_valid_wls_lazy(self, props) -> sp.Expr:
		return ct.LazyArrayRangeFlow(
			start=props['wl_range'][0] / spu.nm,
			stop=props['wl_range'][0] / spu.nm,
			steps=0,
			scaling=ct.ScalingMode.Lin,
			unit=spu.nm,
		)

	####################
	# - Preview
	####################
	@events.on_show_plot(
		managed_objs={'plot'},
		props={'medium'},
		stop_propagation=True,
	)
	def on_show_plot(
		self,
		managed_objs,
		props,
	):
		managed_objs['plot'].mpl_plot_to_image(
			lambda ax: self.medium.plot(props['medium'].frequency_range, ax=ax),
			bl_select=True,
		)
		## TODO: Plot based on Wl, not freq.


####################
# - Blender Registration
####################
BL_REGISTER = [
	LibraryMediumNode,
]
BL_NODES = {ct.NodeType.LibraryMedium: (ct.NodeCategory.MAXWELLSIM_MEDIUMS)}
