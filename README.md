# Node Design
Now that we can do all the cool things ex. presets and such, it's time to think more design.

## Nodes
**NOTE**: Throughout, when an object can be selected (ex. for GeoNodes structure to affect), a button should be available to generate a new object for the occasion.

**NOTE**: Throughout, all nodes that output floats/vectors should have a sympy dimension. Any node that takes floats/vectors should either have a pre-defined unit (exposed as a string in the node UI), or a selectable unit (ex. for value inputs).

- Inputs
	- Scene
		- Time
		- Unit System
	
	- Parameters: Sympy variables.
		- *type* Parameter
	- Constants: Typed numbers.
		- Scientific Constant
		
		- *type* Constant
	- Lists
		- *type* List Element
	
	- File Data: Data from a file.
- Outputs
	- Viewers
		- Value Viewer: Live-monitoring.
		- Console Viewer: w/Button to Print Types
	- Exporters
		- JSON File Exporter: Compatible with any socket implementing `.as_json()`.
	- Plotters
		- *various kinds of plotting? To Blender datablocks primarily, maybe*.

- Sources
	- **ALL**: Accept a Temporal Shape
	
	- Temporal Shapes
		- Gaussian Pulse Temporal Shape
		- Continuous Wave Temporal Shape
		- Array Temporal Shape
	
	- Point Dipole Source
	- Uniform Current Source
	- Plane Wave Source
	- Mode Source
	- Gaussian Beam Source
	- Astigmatic Gaussian Beam Source
	- TFSF Source
	
	- E/H Equivalence Array Source
	- E/H Array Source



- Mediums
	- **ALL**: Accept spatial field. Else, spatial uniformity.
	- **ALL**: Accept non-linearity. Else, linear.
	- **ALL**: Accept space-time modulation. Else, static.
	
	- Library Medium
		- **NOTE**: Should provide an EnumProperty of materials with its own categorizations. It should provide another EnumProperty to choose the experiment. It should also be filterable by wavelength range, maybe also model info. Finally, a reference should be generated on use as text.
	
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
	- Object Structure
	- GeoNodes Structure
	- Scripted Structure
	
	- Primitives
		- Box Structure
		- Sphere Structure
		- Cylinder Structure



- Bounds
	- Bound Box
	
	- Bound Faces
		- PML Bound Face: "Normal"/"Stable"
		- PEC Bound Face
		- PMC Bound Face
		
		- Bloch Bound Face
		- Periodic Bound Face
		- Absorbing Bound Face
	

- Monitors
	- **ALL**: "Steady-State" / "Time Domain" (only if relevant).
	
	- E/H Field Monitor: "Steady-State"
	- Field Power Flux Monitor
	- \epsilon Tensor Monitor
	- Diffraction Monitor
	
	- **TODO**: "Modal" solver monitoring (seems to be some kind of spatial+frequency feature, which an EM field can be decomposed into using a specially configured solver, which can be used to look for very particular kinds of effects by constraining investigations of a solver result to filter out everything that isn't these particular modes aka. features. Kind of a fourier-based redimensionalization, almost).
	- **TODO**: Near-field projections like so:
		- Cartesian Near-Field Projection Monitor
		- Observation Angle Near-Field Projection Monitor
		- K-Space Near-Field Projection Monitor



- Simulations
	- Sim Grid
	- Sim Grid Axis
		- Automatic Sim Grid Axis
		- Manual Sim Grid Axis
		- Uniform Sim Grid Axis
		- Array Sim Grid Axis
	
	- FDTD Sim



- Utilities
	- Math: Contains a dropdown for operation.
		- *type* Math: **Be careful about units :)**
	- Operations
		- List Operation

## Sockets
- basic
	- Any
	- FilePath
	- Text
- number
	- IntegerNumber
	- RationalNumber
	
	- RealNumber
	- ComplexNumber
	- RealNumberField
	- ComplexNumberField
	
- vector
	- Real2DVector
	- Complex2DVector
	- Real2DVectorField
	- Complex2DVectorField
	
	- Real3DVector
	- Complex3DVector
	- Real3DVectorField
	- Complex3DVectorField
- physics
	- PhysicalTime
	
	- PhysicalAngle
	
	- PhysicalLength
	- PhysicalArea
	- PhysicalVolume
	
	- PhysicalMass
	- PhysicalLengthDensity
	- PhysicalAreaDensity
	- PhysicalVolumeDensity
	
	- PhysicalSpeed
	- PhysicalAcceleration
	- PhysicalForce
	
	- PhysicalPolarization
	
	- PhysicalFrequency
	- PhysicalSpectralDistribution
- blender
	- BlenderObject
	- BlenderCollection
	
	- BlenderGeoNodes
	- BlenderImage
- maxwell
	- MaxwellMedium
	- MaxwellMediumNonLinearity
	
	- MaxwellStructure
	
	- MaxwellBoundBox
	- MaxwellBoundFace
	
	- MaxwellMonitor
	
	- MaxwellSimGrid
	
	- FDTDSim



### GeoNode Trees
For ease of use, we can ship with premade node trees/groups for:
- Primitives
	- Plane
	- Box
	- Sphere
	- Cylinder
	- Ring
	- Capsule
	- Cone
- Array
	- Square Array: Takes a primitive shape.
	- Hex Array: Takes a primitive shape.
- Hole Array
	- Square Hole Array: Takes a primitive hole shape.
	- Hex Hole Array: Takes a primitive hole shape.
- Cavities
	- Hex Array w/ L-Cavity: Takes a primitive hole shape.
	- Hex Array w/ H-Cavity: Takes a primitive hole shape.
- Crystal
	- FCC Sphere Array: Takes a primitive spherical-like shape.
	- BCC Sphere Array: Takes a primitive spherical-like shape.
- Wood Pile

When it comes to geometry, we do need to make sure

### Notes
**NOTE**: When several geometries assigned to the same medium are assigned to the same `tidy3d.GeometryGroup`, there can apparently be "significant performance enhancement"s (<https://docs.flexcompute.com/projects/tidy3d/en/latest/_autosummary/tidy3d.GeometryGroup.html#tidy3d.GeometryGroup>).
- We can and should, in the Simulation builder (or just the structure concatenator), batch together structures with the same Medium.

**NOTE**: Some symmetries can be greatly important for performance. <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/Symmetry.html>
