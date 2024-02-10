# Node Design
Now that we can do all the cool things ex. presets and such, it's time to think more design.

## Categories
**NOTE**: Throughout, when an object can be selected (ex. for GeoNodes structure to affect), a button should be available to generate a new object for the occasion.

**NOTE**: Throughout, all nodes that output floats/vectors should have a sympy dimension. Any node that takes floats/vectors should either have a pre-defined unit (exposed as a string in the node UI), or a selectable unit (ex. for value inputs).

- Inputs
	- Scene
		- Time
		- Object Info
	- Parameter
		- Float Parameter
		- Complex Parameter
		- Vec3 Parameter
	- Constant
		- Scientific Constant
		- Float Constant
		- Complex Constant
		- 3-Vector Constant
	- Array
		- Element: Create a 1-element array, with a typed value.
			- Float Array Element
			- Complex Array Element
			- 3-Vector Array Element
		- Union: Concatenate two arrays.
			- Float Array Union
			- Complex Array Union
			- 3-Vector Array Union
	- Dictionary
		- Element: Create a 1-element (typed) dictionary, with a string and a typed value.
			- Float Dict Element
			- Complex Dict Element
			- 3-Vector Dict Element
		- Union: Concatenate two dictionaries.
			- Float Dict Element
			- Complex Dict Element
			- 3-Vector Dict Element
	- Field
		- Float Field
		- Complex Field
		- Vec3 Field

- Outputs
	- Viewer
		- Value Viewer: Live-monitor non-special types.
		- Console Viewer: Print to console with button.
	- Exporter
		- JSON File Export

- Viz
	- Temporal Shape Viz: Some kind of plot (animated) of the shape.
	- Source Viz: 3D 
	- Structure Viz
	- Bound Viz
	- FDTD Viz



- Sources
	- Temporal Shapes
		- Gaussian Pulse Temporal Shape
		- Continuous Wave Temporal Shape
		- Data Driven Temporal Shape
	
	- Modelled
		- Point Dipole Source
		- Uniform Current Source
		- Plane Wave Source
		- Mode Source
		- Gaussian Beam Source
		- Astigmatic Gaussian Beam Source
		- TFSF Source
	
	- Data-Driven
		- E/H Equivalence Source
		- E/H Source



- Mediums
	- **NOTE**: Mediums should optionally accept a spatially varying field. If not, the medium should be considered spatially uniform.
	- **NOTE**: Mediums should optionally accept non-linear effects, either individually or summed using Non-Linear / Operations / Add.
	- **NOTE**: Mediums should optionally accept space-time modulation effects, either individually or summed using Non-Linear / Operations / Add.
	
	- Library Medium
		- **NOTE**: Should provide an EnumProperty of materials with its own categorizations. It should provide another EnumProperty to choose the experiment. It should also be filterable by wavelength range, maybe also model info. Finally, a reference should be generated on use as text.
	
	- Linear Mediums
		- PEC Medium
		- Isotropic Medium
		- Anisotropic Medium
		
		- 3-Sellmeier Medium
		- Sellmeier Medium
		- Pole-Residue Medium
		- Drude Medium
		- Drude-Lorentz Medium
		- Debye Medium
	
	- Non-Linearities
		- Add Non-Linearity
		- \chi_3 Susceptibility Non-Linearity
		- Two-Photon Absorption Non-Linearity
		- Kerr Non-Linearity
	
	- Space/Time \epsilon/\mu Modulation

- Structures
	- TriMesh Structure
	
	- Primitives
		- Box Structure
		- Sphere Structure
		- Cylinder Structure
	- Generated
		- GeoNodes Structure: Takes dict, compute geonode tree on new object.
		- Scripted Structure: Python script generating geometry.



- Bounds
	- Bound Box
	
	- Bound Faces
		- PML Bound Face: Should have an option to switch to "stable".
		- PEC Bound Face
		- PMC Bound Face
		
		- Bloch Bound Face
		- Periodic Bound Face
		- Absorbing Bound Face
	

- Monitors
	- **NOTE**: Some/all should have dropdown to choose between a single (aka. steady-state) monitoring at the end of the simulation, or a continuous / animated (aka. movie) monitoring over the course of the whole simulation.
	
	- **TODO**: "Modal" solver monitoring (seems to be some kind of spatial+frequency feature, which an EM field can be decomposed into using a specially configured solver, which can be used to look for very particular kinds of effects by constraining investigations of a solver result to filter out everything that isn't these particular modes aka. features. Kind of a fourier-based redimensionalization, almost).
	
	- E/H Field Monitor
	- Field Power Flux Monitor
	- \epsilon Tensor Monitor
	- Diffraction Monitor
	
	- Near-Field Projections
		- **TODO**: Figure out exactly how to deal with these visually.
		
		- Cartesian Near-Field Projection Monitor
		- Observation Angle Near-Field Projection Monitor
		- K-Space Near-Field Projection Monitor



- Simulations
	- FDTD Simulation
	
	- Discretizations
		- Simulation Grid Discretization
		- 1D Grid Discretizations
			- Automatic 1D Grid Discretization
			- Manual 1D Grid Discretization
			- Uniform 1D Grid Discretization
			- Data-Driven 1D Grid Discretization



- Utilities
	- Math: Contains a dropdown for operation.
		- Float Math
		- Complex Math
		- Vector Math
	- Field Math: Contains a dropdown for operation.
		- Float Field Math
		- Complex Field Math
		- 3-Vector Field Math
	- Spectral Math: Contains a dropdown for operation.


### Structures

All should support
- Medium

TriMesh should support:
- EnumProperty defining whether to select an object or a collection.
- Medium

**NOTE**: When several geometries assigned to the same medium are assigned to the same `tidy3d.GeometryGroup`, there can apparently be "significant performance enhancement"s (<https://docs.flexcompute.com/projects/tidy3d/en/latest/_autosummary/tidy3d.GeometryGroup.html#tidy3d.GeometryGroup>).
- We can and should, in the Simulation builder (or just the structure concatenator), batch together structures with the same Medium.

### 
