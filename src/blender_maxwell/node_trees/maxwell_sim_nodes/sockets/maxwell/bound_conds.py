import bpy
import tidy3d as td

from ... import contracts as ct
from .. import base

BOUND_FACE_ITEMS = [
	('PML', 'PML', 'Perfectly matched layer'),
	('PEC', 'PEC', 'Perfect electrical conductor'),
	('PMC', 'PMC', 'Perfect magnetic conductor'),
	('PERIODIC', 'Periodic', 'Infinitely periodic layer'),
]
BOUND_MAP = {
	'PML': td.PML(),
	'PEC': td.PECBoundary(),
	'PMC': td.PMCBoundary(),
	'PERIODIC': td.Periodic(),
}


class MaxwellBoundCondsBLSocket(base.MaxwellSimSocket):
	socket_type = ct.SocketType.MaxwellBoundConds
	bl_label = 'Maxwell Bound Box'

	####################
	# - Properties
	####################
	show_definition: bpy.props.BoolProperty(
		name='Show Bounds Definition',
		description='Toggle to show bound faces',
		default=False,
		update=(lambda self, context: self.on_prop_changed('show_definition', context)),
	)

	x_pos: bpy.props.EnumProperty(
		name='+x Bound Face',
		description='+x choice of default boundary face',
		items=BOUND_FACE_ITEMS,
		default='PML',
		update=(lambda self, context: self.on_prop_changed('x_pos', context)),
	)
	x_neg: bpy.props.EnumProperty(
		name='-x Bound Face',
		description='-x choice of default boundary face',
		items=BOUND_FACE_ITEMS,
		default='PML',
		update=(lambda self, context: self.on_prop_changed('x_neg', context)),
	)
	y_pos: bpy.props.EnumProperty(
		name='+y Bound Face',
		description='+y choice of default boundary face',
		items=BOUND_FACE_ITEMS,
		default='PML',
		update=(lambda self, context: self.on_prop_changed('y_pos', context)),
	)
	y_neg: bpy.props.EnumProperty(
		name='-y Bound Face',
		description='-y choice of default boundary face',
		items=BOUND_FACE_ITEMS,
		default='PML',
		update=(lambda self, context: self.on_prop_changed('y_neg', context)),
	)
	z_pos: bpy.props.EnumProperty(
		name='+z Bound Face',
		description='+z choice of default boundary face',
		items=BOUND_FACE_ITEMS,
		default='PML',
		update=(lambda self, context: self.on_prop_changed('z_pos', context)),
	)
	z_neg: bpy.props.EnumProperty(
		name='-z Bound Face',
		description='-z choice of default boundary face',
		items=BOUND_FACE_ITEMS,
		default='PML',
		update=(lambda self, context: self.on_prop_changed('z_neg', context)),
	)

	####################
	# - UI
	####################
	def draw_label_row(self, row: bpy.types.UILayout, text) -> None:
		row.label(text=text)
		row.prop(self, 'show_definition', toggle=True, text='', icon='MOD_LENGTH')

	def draw_value(self, col: bpy.types.UILayout) -> None:
		if not self.show_definition:
			return

		for axis in ['x', 'y', 'z']:
			row = col.row(align=False)
			split = row.split(factor=0.2, align=False)

			_col = split.column(align=True)
			_col.alignment = 'RIGHT'
			_col.label(text=axis + ' -')
			_col.label(text=' +')

			_col = split.column(align=True)
			_col.prop(self, axis + '_neg', text='')
			_col.prop(self, axis + '_pos', text='')

	####################
	# - Computation of Default Value
	####################
	@property
	def value(self) -> td.BoundarySpec:
		return td.BoundarySpec(
			x=td.Boundary(
				plus=BOUND_MAP[self.x_pos],
				minus=BOUND_MAP[self.x_neg],
			),
			y=td.Boundary(
				plus=BOUND_MAP[self.y_pos],
				minus=BOUND_MAP[self.y_neg],
			),
			z=td.Boundary(
				plus=BOUND_MAP[self.z_pos],
				minus=BOUND_MAP[self.z_neg],
			),
		)


####################
# - Socket Configuration
####################
class MaxwellBoundCondsSocketDef(base.SocketDef):
	socket_type: ct.SocketType = ct.SocketType.MaxwellBoundConds

	def init(self, bl_socket: MaxwellBoundCondsBLSocket) -> None:
		pass


####################
# - Blender Registration
####################
BL_REGISTER = [
	MaxwellBoundCondsBLSocket,
]
