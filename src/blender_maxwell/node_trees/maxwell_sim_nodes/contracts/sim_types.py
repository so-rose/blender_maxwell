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

"""Declares various simulation types for use by nodes and sockets."""

import dataclasses
import enum
import functools
import typing as typ
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import jaxtyping as jtyp
import numpy as np
import polars as pl
import pydantic as pyd
import tidy3d as td

from blender_maxwell.contracts import BLEnumElement
from blender_maxwell.services import tdcloud
from blender_maxwell.utils import logger, sim_symbols
from blender_maxwell.utils import sympy_extra as spux

from .flow_kinds.info import InfoFlow

log = logger.get(__name__)


####################
# - JAX-Helpers
####################
def manual_amp_time(self, time: float) -> complex:
	"""Copied implementation of `pulse.amp_time` for `tidy3d` temporal shapes, which replaces use of `numpy` with `jax.numpy` for `jit`-ability.

	Since the function is detached from the method, `self` is not implicitly available. It should be pre-defined from a real source time object using `functools.partial`, before `jax.jit`ing.

	## License
	**This function is directly copied from `tidy3d`**.
	As such, it should be considered available under the `tidy3d` license (as of writing, LGPL 2.1): <https://github.com/flexcompute/tidy3d/blob/develop/LICENSE>

	## Reference
	Permalink to GitHub source code: <https://github.com/flexcompute/tidy3d/blob/3ee34904eb6687a86a5fb3f4ed6d3295c228cd83/tidy3d/components/source.py#L143C1-L163C25>
	"""
	twidth = 1.0 / (2 * jnp.pi * self.fwidth)
	omega0 = 2 * jnp.pi * self.freq0
	time_shifted = time - self.offset * twidth

	offset = jnp.exp(1j * self.phase)
	oscillation = jnp.exp(-1j * omega0 * time)
	amp = jnp.exp(-(time_shifted**2) / 2 / twidth**2) * self.amplitude

	pulse_amp = offset * oscillation * amp

	# subtract out DC component
	if self.remove_dc_component:
		pulse_amp = pulse_amp * (1j + time_shifted / twidth**2 / omega0)
	else:
		# 1j to make it agree in large omega0 limit
		pulse_amp = pulse_amp * 1j

	return pulse_amp


## TODO: Sim Domain type, w/pydantic checks!


####################
# - Global Simulation Coordinate System
####################
class SimSpaceAxis(enum.StrEnum):
	"""The axis labels of the global simulation coordinate system."""

	X = enum.auto()
	Y = enum.auto()
	Z = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		SSA = SimSpaceAxis
		return {
			SSA.X: 'x',
			SSA.Y: 'y',
			SSA.Z: 'z',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	@property
	def axis(self) -> int:
		"""Deduce the integer index of the axis.

		Returns:
			The integer index of the axis.
		"""
		SSA = SimSpaceAxis
		return {SSA.X: 0, SSA.Y: 1, SSA.Z: 2}[self]


class SimAxisDir(enum.StrEnum):
	"""Positive or negative direction along an injection axis."""

	Plus = enum.auto()
	Minus = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		SAD = SimAxisDir
		return {
			SAD.Plus: '+',
			SAD.Minus: '-',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	@property
	def plus_or_minus(self) -> int:
		"""Get '+' or '-' literal corresponding to the direction.

		Returns:
			The appropriate literal.
		"""
		SAD = SimAxisDir
		return {SAD.Plus: '+', SAD.Minus: '-'}[self]

	@property
	def true_or_false(self) -> bool:
		"""Get 'True' or 'False' bool corresponding to the direction.

		Returns:
			The appropriate bool.
		"""
		SAD = SimAxisDir
		return {SAD.Plus: True, SAD.Minus: False}[self]


####################
# - Simulation Fields
####################
class SimFieldPols(enum.StrEnum):
	"""Positive or negative direction along an injection axis."""

	Ex = 'Ex'
	Ey = 'Ey'
	Ez = 'Ez'
	Hx = 'Hx'
	Hy = 'Hy'
	Hz = 'Hz'

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		SFP = SimFieldPols
		return {
			SFP.Ex: 'Ex',
			SFP.Ey: 'Ey',
			SFP.Ez: 'Ez',
			SFP.Hx: 'Hx',
			SFP.Hy: 'Hy',
			SFP.Hz: 'Hz',
		}[v]

	@property
	def name(self) -> str:
		return SimFieldPols.to_name(self)

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''


####################
# - Boundary Condition Type
####################
class BoundCondType(enum.StrEnum):
	r"""A type of boundary condition, applied to a half-axis of a simulation domain.

	Attributes:
		Pml: "Perfectly Matched Layer" models infinite free space.
			**Should be placed sufficiently far** (ex. $\frac{\lambda}{2}) from any active structures to mitigate divergence.
		Periodic: Denotes naive Bloch boundaries (aka. periodic w/phase shift of 0).
		Pec: "Perfect Electrical Conductor" models a surface that perfectly reflects electric fields.
		Pmc: "Perfect Magnetic Conductor" models a surface that perfectly reflects the magnetic fields.
	"""

	Pml = enum.auto()
	NaiveBloch = enum.auto()
	Pec = enum.auto()
	Pmc = enum.auto()

	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""Convert the enum value to a human-friendly name.

		Notes:
			Used to print names in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		BCT = BoundCondType
		return {
			BCT.Pml: 'PML',
			BCT.Pec: 'PEC',
			BCT.Pmc: 'PMC',
			BCT.NaiveBloch: 'NaiveBloch',
		}[v]

	@staticmethod
	def to_icon(_: typ.Self) -> str:
		"""Convert the enum value to a Blender icon.

		Notes:
			Used to print icons in `EnumProperty`s based on this enum.

		Returns:
			A human-friendly name corresponding to the enum value.
		"""
		return ''

	@property
	def tidy3d_boundary_edge(self) -> td.BoundaryEdge:
		"""Convert the boundary condition specifier to a corresponding, sensible `tidy3d` boundary edge.

		`td.BoundaryEdge` can be used to declare a half-axis in a `td.BoundarySpec`, which attaches directly to a simulation object.

		Returns:
			A sensible choice of `tidy3d` object representing the boundary condition.
		"""
		BCT = BoundCondType
		return {
			BCT.Pml: td.PML(),
			BCT.Pec: td.PECBoundary(),
			BCT.Pmc: td.PMCBoundary(),
			BCT.NaiveBloch: td.Periodic(),
		}[self]


####################
# - Cloud Task
####################
@dataclasses.dataclass(kw_only=True, frozen=True)
class NewSimCloudTask:
	"""Not-yet-existing simulation-oriented cloud task."""

	task_name: tdcloud.CloudTaskName
	cloud_folder: tdcloud.CloudFolder


####################
# - Data File
####################
_DATA_FILE_EXTS = {
	'.txt',
	'.txt.gz',
	'.csv',
	'.npy',
}


class DataFileFormat(enum.StrEnum):
	"""Abstraction of a data file format, providing a regularized way of interacting with filesystem data.

	Import/export interacts closely with the `Expr` socket's `FlowKind` semantics:
	- `FlowKind.Func`: Generally realized on-import/export.
		- **Import**: Loading data is generally eager, but memory-mapped file loading would be manageable using this interface.
		- **Export**: The function is realized and only the array is inserted into the file.
	- `FlowKind.Params`: Generally consumed.
		- **Import**: A new, empty `ParamsFlow` object is created.
		- **Export**: The `ParamsFlow` is consumed when realizing the `Func`.
	- `FlowKind.Info`: As the most important element, it is kept in an (optional) sidecar metadata file.
		- **Import**: The sidecar file is loaded, checked, and used, if it exists. A warning about further processing may show if it doesn't.
		- **Export**: The sidecar file is written next to the canonical data file, in such a manner that it can be both read and loaded.

	Notes:
		This enum is UI Compatible, ex. for nodes/sockets desiring a dropdown menu of data file formats.

	Attributes:
		Txt: Simple no-header text file.
			Only supports 1D/2D data.
		TxtGz: Identical to `Txt`, but compressed with `gzip`.
		Csv: Unspecific "Comma Separated Values".
			For loading, `pandas`-default semantics are used.
			For saving, very opinionated defaults are used.
			Customization is disabled on purpose.
		Npy: Generic numpy representation.
			Supports all kinds of numpy objects.
			Better laziness support via `jax`.
	"""

	Csv = enum.auto()
	Npy = enum.auto()
	Txt = enum.auto()
	TxtGz = enum.auto()

	####################
	# - UI
	####################
	@staticmethod
	def to_name(v: typ.Self) -> str:
		"""The extension name of the given `DataFileFormat`.

		Notes:
			Called by the UI when creating an `EnumProperty` dropdown.
		"""
		return DataFileFormat(v).extension

	@staticmethod
	def to_icon(v: typ.Self) -> str:
		"""No icon.

		Notes:
			Called by the UI when creating an `EnumProperty` dropdown.
		"""
		return ''

	def bl_enum_element(self, i: int) -> BLEnumElement:
		"""Produce a fully functional Blender enum element, given a particular integer index."""
		return (
			str(self),
			DataFileFormat.to_name(self),
			DataFileFormat.to_name(self),
			DataFileFormat.to_icon(self),
			i,
		)

	@staticmethod
	def bl_enum_elements() -> list[BLEnumElement]:
		"""Produce an immediately usable list of Blender enum elements, correctly indexed."""
		return [
			data_file_format.bl_enum_element(i)
			for i, data_file_format in enumerate(list(DataFileFormat))
		]

	####################
	# - Properties
	####################
	@property
	def extension(self) -> str:
		"""Map to the actual string extension."""
		E = DataFileFormat
		return {
			E.Csv: '.csv',
			E.Npy: '.npy',
			E.Txt: '.txt',
			E.TxtGz: '.txt.gz',
		}[self]

	####################
	# - Creation: Compatibility
	####################
	@staticmethod
	def valid_exts() -> list[str]:
		return _DATA_FILE_EXTS

	@staticmethod
	def ext_has_valid_format(ext: str) -> bool:
		return ext in _DATA_FILE_EXTS

	@staticmethod
	def path_has_valid_format(path: Path) -> bool:
		return path.is_file() and DataFileFormat.ext_has_valid_format(
			''.join(path.suffixes)
		)

	def is_path_compatible(
		self, path: Path, must_exist: bool = False, can_exist: bool = True
	) -> bool:
		ext_matches = self.extension == ''.join(path.suffixes)
		match (must_exist, can_exist):
			case (False, False):
				return ext_matches and not path.is_file() and path.parent.is_dir()

			case (True, False):
				msg = f'DataFileFormat: Path {path} cannot both be required to exist (must_exist=True), but also not be allowed to exist (can_exist=False)'
				raise ValueError(msg)

			case (False, True):
				return ext_matches and path.parent.is_dir()

			case (True, True):
				return ext_matches and path.is_file()

	####################
	# - Creation
	####################
	@staticmethod
	def from_ext(ext: str) -> typ.Self | None:
		return {
			_ext: _data_file_ext
			for _data_file_ext, _ext in {
				k: k.extension for k in list(DataFileFormat)
			}.items()
		}.get(ext)

	@staticmethod
	def from_path(path: Path) -> typ.Self | None:
		if DataFileFormat.path_has_valid_format(path):
			data_file_ext = DataFileFormat.from_ext(''.join(path.suffixes))
			if data_file_ext is not None:
				return data_file_ext

			msg = f'DataFileFormat: Path "{path}" is compatible, but could not find valid extension'
			raise RuntimeError(msg)

		return None

	####################
	# - Functions: Metadata
	####################
	def supports_metadata(self) -> bool:
		E = DataFileFormat
		return {
			E.Csv: False,  ## No RFC 4180 Support for Comments
			E.Npy: False,  ## Quite simply no support
			E.Txt: True,  ## Use # Comments
			E.TxtGz: True,  ## Same as Txt
		}[self]

	## TODO: Sidecar Metadata
	## - The vision is that 'saver' also writes metadata.
	## - This metadata is essentially a straight serialization of the InfoFlow.
	## - On-load, the metadata is used to re-generate the InfoFlow.
	## - This allows interpreting saved data without a ton of shenanigans.
	## - These sidecars could also be hand-writable for external data.
	## - When sidecars aren't found, the user would "fill in the blanks".
	## - ...Thus achieving the same result as if there were a sidecar.

	####################
	# - Functions: DataFrame
	####################
	@staticmethod
	def to_df(
		data: jtyp.Shaped[jtyp.Array, 'x_size y_size'], info: InfoFlow
	) -> pl.DataFrame:
		"""Utility method to convert raw data to a `polars.DataFrame`, as guided by an `InfoFlow`.

		Only works with 2D data (obviously).

		Raises:
			ValueError: If the data has more than two dimensions, all `info` dimensions are not discrete/labelled, or the dimensionality of `info` doesn't match.
		"""
		if info.order > 2:  # noqa: PLR2004
			msg = f'Data may not have more than two dimensions (info={info}, data.shape={data.shape})'
			raise ValueError(msg)

		if any(info.has_idx_cont(dim) for dim in info.dims):
			msg = f'To convert data|info to a dataframe, no dimensions can have continuous indices (info={info})'
			raise ValueError(msg)

		data_np = np.array(data)

		MT = spux.MathType
		match (
			info.input_mathtypes,
			info.output.mathtype,
			info.output.rows,
			info.output.cols,
		):
			# (R,Z) -> Complex Scalar
			## -> Polars (also pandas) doesn't have a complex type.
			## -> Will be treated as (R, Z, 2) -> Real Scalar.
			case ((MT.Rational | MT.Real, MT.Integer), MT.Complex, 1, 1):
				row_dim = info.first_dim
				col_dim = info.last_dim

				return pl.DataFrame(
					{row_dim.name: info.dims[row_dim]}
					| {
						col_label + postfix: re_im(data_np[:, col])
						for col, col_label in enumerate(info.dims[col_dim])
						for postfix, re_im in [('_re', np.real), ('_im', np.imag)]
					}
				)

			# (R,Z) -> Scalar
			case ((MT.Rational | MT.Real, MT.Integer), _, 1, 1):
				row_dim = info.first_dim
				col_dim = info.last_dim

				return pl.DataFrame(
					{row_dim.name: info.dims[row_dim]}
					| {
						col_label: data_np[:, col]
						for col, col_label in enumerate(info.dims[col_dim])
					}
				)

			# (Z) -> Complex Vector/Covector
			case ((MT.Integer,), MT.Complex, r, c) if (r > 1 and c == 1) or (
				r == 1 and c > 1
			):
				col_dim = info.last_dim

				return pl.DataFrame(
					{
						col_label + postfix: re_im(data_np[col, :])
						for col, col_label in enumerate(info.dims[col_dim])
						for postfix, re_im in [('_re', np.real), ('_im', np.imag)]
					}
				)

			# (Z) -> Real Vector
			## -> Each integer index will be treated as a column index.
			## -> This will effectively transpose the data.
			case ((MT.Integer,), _, r, c) if (r > 1 and c == 1) or (r == 1 and c > 1):
				col_dim = info.last_dim

				return pl.DataFrame(
					{
						col_label: data_np[col, :]
						for col, col_label in enumerate(info.dims[col_dim])
					}
				)

	####################
	# - Functions: Saver
	####################
	def is_info_compatible(self, info: InfoFlow) -> bool:
		E = DataFileFormat
		match self:
			case E.Csv:
				return len(info.dims) + info.output.rows + info.output.cols - 1 <= 2
			case E.Npy:
				return True
			case E.Txt | E.TxtGz:
				return len(info.dims) + info.output.rows + info.output.cols - 1 <= 2

	@property
	def saver(
		self,
	) -> typ.Callable[[Path, jtyp.Shaped[jtyp.Array, '...'], InfoFlow], None]:
		def save_txt(path, data, info):
			np.savetxt(path, data)

		def save_txt_gz(path, data, info):
			np.savetxt(path, data)

		def save_csv(path, data, info):
			df = self.to_df(data, info)
			df.write_csv(path)

		def save_npy(path, data, info):
			jnp.save(path, data)

		E = DataFileFormat
		return {
			E.Csv: save_csv,
			E.Npy: save_npy,
			E.Txt: save_txt,
			E.TxtGz: save_txt_gz,
		}[self]

	####################
	# - Functions: Loader
	####################
	@property
	def loader_is_jax_compatible(self) -> bool:
		E = DataFileFormat
		return {
			E.Csv: False,
			E.Npy: True,
			E.Txt: True,
			E.TxtGz: True,
		}[self]

	@property
	def loader(
		self,
	) -> typ.Callable[[Path], tuple[jtyp.Shaped[jtyp.Array, '...'], InfoFlow]]:
		def load_txt(path: Path):
			return jnp.asarray(np.loadtxt(path))

		def load_csv(path: Path):
			return jnp.asarray(pl.read_csv(path).to_numpy())
			## TODO: The very next Polars (0.20.27) has a '.to_jax' method!

		def load_npy(path: Path):
			return jnp.load(path)

		E = DataFileFormat
		return {
			E.Csv: load_csv,
			E.Npy: load_npy,
			E.Txt: load_txt,
			E.TxtGz: load_txt,
		}[self]

	####################
	# - Metadata: Compatibility
	####################
	def is_info_compatible(self, info: InfoFlow) -> bool:
		E = DataFileFormat
		match self:
			case E.Csv:
				return len(info.dims) + (info.output.rows + input.outputs.cols - 1) <= 2
			case E.Npy:
				return True
			case E.Txt | E.TxtGz:
				return len(info.dims) + (info.output.rows + info.output.cols - 1) <= 2

	def supports_metadata(self) -> bool:
		E = DataFileFormat
		return {
			E.Csv: False,  ## No RFC 4180 Support for Comments
			E.Npy: False,  ## Quite simply no support
			E.Txt: True,  ## Use # Comments
			E.TxtGz: True,  ## Same as Txt
		}[self]


####################
# - Encode/Decode Metadata
####################
RealizationScalar: typ.TypeAlias = int | float
Realization: typ.TypeAlias = (
	RealizationScalar
	| tuple[RealizationScalar, ...]
	| tuple[tuple[RealizationScalar, ...], ...]
)


class SimRealizations(pyd.BaseModel):
	"""Encodes the realized values of symbols that were used to generate a particular simulation."""

	model_config = pyd.ConfigDict(frozen=True)

	syms: tuple[sim_symbols.SimSymbol, ...] = ()
	vals: tuple[Realization, ...] = ()

	@pyd.model_validator(mode='after')
	def syms_vals_eq_length(self) -> typ.Self:
		"""Ensure that `self.syms` and `self.vals` are of equal length."""
		if len(self.syms) != len(self.vals):
			msg = f"'syms' and 'vals' are of differing length (syms={self.syms}, vals={self.vals})"
			raise ValueError(msg)

		return self


class SimMetadata(pyd.BaseModel):
	"""Encodes simulation metadata."""

	model_config = pyd.ConfigDict(frozen=True)

	sim_metadata_version: str = '0.1.0'
	realizations: SimRealizations = SimRealizations()

	@staticmethod
	def from_sim(sim: td.Simulation | td.SimulationData) -> typ.Self:
		"""Deduce simulation metadata from a simulation / simulation data."""
		if 'sim_metadata_version' in sim.attrs:
			## TODO: Semantic versioning comparison
			return SimMetadata(**sim.attrs)
		return SimMetadata()

	@functools.cached_property
	def syms_vals(
		self,
	) -> tuple[tuple[sim_symbols.SimSymbol, ...], tuple[Realization, ...]]:
		"""Deduce simulation metadata from a simulation / simulation data."""
		return (self.realizations.syms, self.realizations.vals)
