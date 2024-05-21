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

from blender_maxwell.utils import blender_type_enum


@blender_type_enum.append_cls_name_to_values
class NodeType(blender_type_enum.BlenderTypeEnum):
	# KitchenSink = enum.auto()

	# Analysis
	ExtractData = enum.auto()
	Viz = enum.auto()
	## Analysis / Math
	OperateMath = enum.auto()
	MapMath = enum.auto()
	FilterMath = enum.auto()
	ReduceMath = enum.auto()
	TransformMath = enum.auto()

	# Inputs
	WaveConstant = enum.auto()
	Scene = enum.auto()
	## Inputs / Constants
	ExprConstant = enum.auto()
	ScientificConstant = enum.auto()
	UnitSystemConstant = enum.auto()
	BlenderConstant = enum.auto()
	## Inputs / Web Importers
	Tidy3DWebImporter = enum.auto()
	## Inputs / File Importers
	DataFileImporter = enum.auto()
	Tidy3DFileImporter = enum.auto()

	# Outputs
	Viewer = enum.auto()
	## Outputs / File Exporters
	DataFileExporter = enum.auto()
	Tidy3DWebExporter = enum.auto()
	## Outputs / Web Exporters
	JSONFileExporter = enum.auto()

	# Sources
	TemporalShape = enum.auto()
	## Sources /
	PointDipoleSource = enum.auto()
	PlaneWaveSource = enum.auto()
	UniformCurrentSource = enum.auto()
	TFSFSource = enum.auto()
	GaussianBeamSource = enum.auto()
	AstigmaticGaussianBeamSource = enum.auto()
	EHDataSource = enum.auto()
	EHEquivDataSource = enum.auto()

	# Mediums
	LibraryMedium = enum.auto()
	DataFitMedium = enum.auto()
	PECMedium = enum.auto()
	IsotropicMedium = enum.auto()
	AnisotropicMedium = enum.auto()
	TripleSellmeierMedium = enum.auto()
	SellmeierMedium = enum.auto()
	PoleResidueMedium = enum.auto()
	DrudeMedium = enum.auto()
	DrudeLorentzMedium = enum.auto()
	DebyeMedium = enum.auto()
	## Mediums / Non-Linearities
	AddNonLinearity = enum.auto()
	ChiThreeSusceptibilityNonLinearity = enum.auto()
	TwoPhotonAbsorptionNonLinearity = enum.auto()
	KerrNonLinearity = enum.auto()

	# Structures
	BLObjectStructure = enum.auto()
	GeoNodesStructure = enum.auto()
	## Structures / Primitives
	LineStructure = enum.auto()
	PlaneStructure = enum.auto()
	BoxStructure = enum.auto()
	SphereStructure = enum.auto()
	CylinderStructure = enum.auto()
	PolySlabStructure = enum.auto()

	# Bounds
	BoundConds = enum.auto()
	## Bounds / Bound Conds
	PMLBoundCond = enum.auto()
	BlochBoundCond = enum.auto()
	AdiabAbsorbBoundCond = enum.auto()

	# Monitors
	EHFieldMonitor = enum.auto()
	PowerFluxMonitor = enum.auto()
	PermittivityMonitor = enum.auto()
	DiffractionMonitor = enum.auto()
	## Monitors / Projected
	CartesianNearFieldProjectionMonitor = enum.auto()
	AngleNearFieldProjectionMonitor = enum.auto()
	KSpaceNearFieldProjectionMonitor = enum.auto()

	# Sims
	Combine = enum.auto()
	SimDomain = enum.auto()
	FDTDSim = enum.auto()
	SimGrid = enum.auto()
	## Sims / Sim Grid Axis
	AutomaticSimGridAxis = enum.auto()
	ManualSimGridAxis = enum.auto()
	UniformSimGridAxis = enum.auto()
	ArraySimGridAxis = enum.auto()

	# Utilities
	Separate = enum.auto()
