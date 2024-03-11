# Nodes
**LEGEND**:
- [-] Exists but doesn't quite work good enough.
- [x] Done to working degree (the standard is "good enough for the demo").
	- See check marks underneath
- [?] Unsure whether we should do this.

## Inputs
[x] Wave Constant
- [ ] Implement export of frequency / wavelength array/range.
[-] Unit System
- [ ] Implement presets, including "Tidy3D" and "Blender", shown in the label row.

[ ] Constants / Scientific Constant
[x] Constants / Number Constant
[ ] Constants / Physical Constant
- [ ] Pol: Elliptical plot viz
- [ ] Pol: Poincare sphere viz
[x] Constants / Blender Constant

[x] Web / Tidy3D Web Importer

[ ] File Import / JSON File Import
- [ ] Dropdown to choose various supported JSON-sourced objects incl. 
[ ] File Import / Tidy3D File Import
- [ ] Implement HDF-based import of Tidy3D-exported object (which includes ex. mesh data and such)
[ ] File Import / Array File Import
- [ ] Standardize 1D and 2D array loading/saving on numpy's savetxt with gzip enabled.
- [ ] Implement datatype dropdown to guide format from disk, prefilled to detected.
- [ ] Implement unit system input to guide conversion from numpy data type.
- [ ] Implement a LazyValue to provide a data path that avoids having to load massive arrays every time always.

## Outputs
[x] Viewer
- [ ] **BIG ONE**: Remove image preview when disabling plots.
- [ ] Either enforce singleton, or find a way to have several viewers at the same time.
- [ ] A setting that live-previews just a value.
- [ ] Pop-up multiline string print as alternative to console print.
- [x] Toggleable auto-plot, auto-3D-preview, auto-value-view, (?)auto-text-view.

[x] File Export / JSON File Export
[ ] File Import / Tidy3D File Export
- [ ] Implement HDF-based export of Tidy3D-exported object (which includes ex. mesh data and such)
[ ] File Export / Array File Export
- [ ] Implement datatype dropdown to guide format on disk.
- [ ] Implement unit system input to guide conversion to numpy data type.
- [ ] Standardize 1D and 2D array loading/saving on numpy's savetxt with gzip enabled.

## Viz
[ ] Monitor Data Viz
- [ ] Implement dropdown to choose which monitor in the SimulationData should be visualized (based on which are available in the SimulationData), and implement visualization based on every kind of monitor-adjascent output data type (<https://docs.flexcompute.com/projects/tidy3d/en/latest/api/output_data.html>)
- [ ] Project field values onto a plane object (managed)

## Sources
[x] Temporal Shapes / Gaussian Pulse Temporal Shape
[x] Temporal Shapes / Continuous Wave Temporal Shape
[ ] Temporal Shapes / Symbolic Temporal Shape
- [ ] Specify a Sympy function to generate appropriate array based on
[ ] Temporal Shapes / Array Temporal Shape

[x] Point Dipole Source
- [ ] Consider a "real" mesh - the empty kind of gets stuck inside of the sim domain.
[-] Plane Wave Source
- [ ] Implement an oriented vector input with 3D preview.
[ ] Uniform Current Source
[ ] TFSF Source

[ ] Gaussian Beam Source
[ ] Astigmatic Gaussian Beam Source

[ ] Mode Source

[ ] Array Source / EH Array Source
[ ] Array Source / EH Equivilance Array Source

## Mediums
[x] Library Medium
- [ ] Implement frequency range output
[ ] PEC Medium
[ ] Isotropic Medium
[ ] Anisotropic Medium

[ ] Sellmeier Medium
[ ] Drude Medium
[ ] Drude-Lorentz Medium
[ ] Debye Medium
[ ] Pole-Residue Medium
	
[ ] Non-Linearity / `chi_3` Susceptibility Non-Linearity
[ ] Non-Linearity / Two-Photon Absorption Non-Linearity
[ ] Non-Linearity / Kerr Non-Linearity

[ ] Space/Time epsilon/mu Modulation

## Structures
[ ] BLObject Structure
[x] GeoNodes Structure
- [x] Rewrite the `bl_socket_map.py`
- [x] Use the modifier itself as memory, via the ManagedObj
- [?] When GeoNodes themselves declare panels, implement a grid-like tab system to select which sockets should be exposed in the node at a given point in time.

[ ] Primitive Structures / Plane
[ ] Primitive Structures / Box Structure
[ ] Primitive Structures / Sphere
[ ] Primitive Structures / Cylinder
[ ] Primitive Structures / Ring
[ ] Primitive Structures / Capsule
[ ] Primitive Structures / Cone

## Monitors
- **ALL**: "Steady-State" / "Time Domain" (only if relevant).

[ ] E/H Field Monitor
- [ ] Monitor Domain as dropdown with Frequency or Time
- [ ] Axis-aligned planar 2D (pixel) and coord-aligned box 3D (voxel).
[ ] Field Power Flux Monitor
- [ ] Monitor Domain as dropdown with Frequency or Time
- [ ] Axis-aligned planar 2D (pixel) and coord-aligned box 3D (voxel).
[ ] \epsilon Tensor Monitor
- [ ] Axis-aligned planar 2D (pixel) and coord-aligned box 3D (voxel).
[ ] Diffraction Monitor
- [ ] Axis-aligned planar 2D (pixel)

[ ] Projected E/H Field Monitor / Cartesian Projected E/H Field Monitor
- [ ] Use to implement the metalens: <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/Metalens.html>
[ ] Projected E/H Field Monitor / Angle Projected E/H Field Monitor
[ ] Projected E/H Field Monitor / K-Space Projected E/H Field Monitor

- **TODO**: "Modal" solver monitoring (seems to be some kind of spatial+frequency feature, which an EM field can be decomposed into using a specially configured solver, which can be used to look for very particular kinds of effects by constraining investigations of a solver result to filter out everything that isn't these particular modes aka. features. Kind of a fourier-based redimensionalization, almost).

## Simulations
[-] FDTDSim

[x] Sim Domain
- [ ] By-Medium batching of Structures when building the td.Simulation object, which can have significant performance implications.

[x] Boundary Conds
- [x] Rename from Bounds / BoundBox
[ ] Boundary Cond / PML Bound Face
- [ ] Implement dropdown for "Normal" and "Stable"
[ ] Boundary Cond / PEC Bound Face
[ ] Boundary Cond / PMC Bound Face
[ ] Boundary Cond / Bloch Bound Face
[ ] Boundary Cond / Periodic Bound Face
[ ] Boundary Cond / Absorbing Bound Face

[ ] Sim Grid
[ ] Sim Grid Axes / Auto Sim Grid Axis
[ ] Sim Grid Axes / Manual Sim Grid Axis
[ ] Sim Grid Axes / Uniform Sim Grid Axis
[ ] Sim Grid Axes / Array Sim Grid Axis

## Converters
[ ] Math
- [ ] Implement common operations w/secondary choice of socket type based on a custom internal data structure
- [ ] Implement angfreq/frequency/vacwl conversion.
[ ] Separate
[ ] Combine
- [ ] Implement concatenation of sim-critical socket types into their multi-type



# GeoNodes
[ ] Tests / Monkey (suzanne deserves to be simulated, she may need manifolding up though :))
[ ] Tests / Wood Pile

[ ] Primitives / Plane
[ ] Primitives / Box
[ ] Primitives / Sphere
[ ] Primitives / Cylinder
[ ] Primitives / Ring
[ ] Primitives / Capsule
[ ] Primitives / Cone

[ ] Array / Square Array **NOTE: Ring and cylinder**
[ ] Array / Hex Array **NOTE: Ring and cylinder**
[ ] Hole Array / Square Hole Array: Takes a primitive hole shape.
[ ] Hole Array / Hex Hole Array: Takes a primitive hole shape.
[ ] Cavity Array / Hex Array w/ L-Cavity
[ ] Cavity Array / Hex Array w/ H-Cavity

[ ] Crystal Sphere Lattice / Sphere FCC Array
[ ] Crystal Sphere Lattice / Sphere BCC Array



# Benchmark / Example Sims
[ ] Research-Grade Experiment
- Membrane 15nm thickness suspended in air
- Square lattice of holes period 900nm (900nm between each hole, air inside holes)
- Holes square radius 100nm
- Square lattice
- Analysis of transmission
- Guided mode resonance
[ ] Tunable Chiral Metasurface <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/TunableChiralMetasurface.html>



# Sockets
## Basic
[x] Any
[x] Bool
[x] String
- [ ] Rename from "Text"
[x] File Path
[x] Color

## Number
[x] Integer
[x] Rational
- [ ] Implement constrained SympyExpr check for Rational.
[x] Real
- [ ] Implement min/max for ex. 0..1 factor support.
- [ ] Implement constrained SympyExpr check for Rational.
[x] Complex

## Blender
[x] Object
- [ ] Implement default SocketDef object name
[x] Collection
- [ ] Implement default SocketDef collection name

[x] Image
- [ ] Implement default SocketDef image name

[x] GeoNodes
- [ ] Implement default SocketDef geonodes name
[x] Text
- [ ] Implement default SocketDef object name

## Maxwell
[x] Bound Conds
[ ] Bound Cond

[x] Medium
[ ] Medium Non-Linearity

[x] Source
[ ] Temporal Shape
- [ ] Sane-default pulses for easy access.

[ ] Structure
[ ] Monitor

[ ] FDTD Sim
[ ] Sim Domain
- [?] Toggleable option to sync the simulation time duration to the scene end time (how to handle FPS vs time-step? Should we adjust the FPS such that there is one time step per frame, while keeping the definition of "second" aligned to a unit system?)
[ ] Sim Grid
[ ] Sim Grid Axis

[ ] Simulation Data

## Tidy3D
[x] Cloud Task
- [ ] Implement switcher for API-key-having config filconfig file vs. direct entry of API key. It should be auto-filled with the config file when such a thing exists.

## Physical
[x] Unit System
- [ ] Implement more comprehensible UI; honestly, probably with the new panels (<https://developer.blender.org/docs/release_notes/4.1/python_api/>)

[x] Time

[x] Angle
[ ] Solid Angle (steradian)

[x] Frequency (hertz)
[ ] Angular Frequency (`rad*hertz`)
### Cartesian
[x] Length
[x] Area
[x] Volume

[ ] Point 1D
[ ] Point 2D
[x] Point 3D

[ ] Size 2D
[x] Size 3D

[ ] Rotation 3D
- [ ] Implement Euler methods
- [ ] Implement Quaternion methods
### Mechanical
[ ] Mass

[x] Speed
[ ] Velocity 3D
[x] Acceleration Scalar
[ ] Acceleration 3D
[x] Force Scalar
[ ] Force 3D
[ ] Pressure
### Energy
[ ] Energy (joule)
[ ] Power (watt)
[ ] Temperature
### Electrodynamical
[ ] Current (ampere)
[ ] Current Density 3D

[ ] Charge (coulomb)
[ ] Voltage (volts)
[ ] Capacitance (farad)
[ ] Resistance (ohm)
[ ] Electric Conductance (siemens)

[ ] Magnetic Flux (weber)
[ ] Magnetic Flux Density (tesla)
[ ] Inductance (henry)

[ ] Electric Field 3D (`volt*meter`)
[ ] Magnetic Field 3D (tesla)
### Luminal
[ ] Luminous Intensity (candela)
[ ] Luminous Flux (lumen)
[ ] Illuminance (lux)
### Optical
[ ] Jones Polarization
[ ] Polarization (Stokes)



# Style
[ ] Rethink the meaning of color and shapes in node sockets, including whether dynamic functionality is needed when it comes to socket shape (ex. it might be nice to know whether a socket is array-like or uses units).
[ ] Rethink the meaning of color and shapes in node sockets, including whether dynamic functionality is needed when it comes to socket shape.



# Architecture
## Registration and Contracts
[x] Finish the contract code converting from Blender sockets to our sockets based on dimensionality and the property description.
[ ] Refactor the node category code; it's ugly.
[?] Would be nice with some kind of indicator somewhere to help set good socket descriptions when using geonodes and wanting units.

## Managed Objects
[x] Implement modifier support on the managed BL object, with special attention paid to the needs of the GeoNodes socket.
- [x] Implement preview toggling too, ex. using the relevant node tree collections
- Remember, the managed object is "dumb". It's the node's responsibility to react to any relevant `on_value_change`, and forward all state needed by the modifier to the managed obj. It's only the managed obj's responsibility to not update any modifier value that wouldn't change anything.
[ ] Implement loading the xarray-defined voxels into OpenVDB, saving it, and loading it as a managed BL object with the volume setting.
[ ] Implement basic jax-driven volume voxel processing, especially cube based slicing.
[ ] Implement jax-driven linear interpolation of volume voxels to an image texture, whose pixels are sized according to the dimensions of another managed plane object (perhaps a uniquely described Managed BL object itself).

## Utils or Services
[ ] Dedicated module for managing the interaction with the tidy3d cloud, to help nuke all the random caches out of existance.

## Node Base Class
[ ] Dedicated `draw_preview`-type draw functions for plot customizations.
- [ ] For now, previewing isn't something I think should be part of the node
[ ] Custom `@cache`/`@lru_cache`/`@cached_property` which caches by instance ID (possibly based on `beartype` or `pydantic`).
[ ] When presets are used, if a preset is selected and the user alters a preset setting, then dynamically switch the preset indicator back to "Custom"  to indicate that there is no active preset
[ ] It seems that `node.inputs` and `node.outputs` allows the use of a `move` method, which may allow reordering sockets dynamically, which we should expose to the user as user-configurable ordering rules (maybe resolved with a constraint solver).
[?] Mechanism for dynamic names (ex. "Library Medium" becoming "Au Medium")
[-] Mechanism for selecting a blender object managed by a particular node.
[ ] Mechanism for ex. specially coloring a node that is currently participating in the preview.
[ ] Custom callbacks when deleting a node (in `free()`), to ex. delete all previews with the viewer node.

## Socket Base Class
[ ] A feature `use_array` which allows a socket to declare that it can be both a single value and array-like (possibly constrained to a given shape). This should also allow the SocketDef to request that the input socket be initialised as a multi-input socket, once Blender updates to support those.
- [ ] Implement a shape-selector, with a dropdown for dimensionality and an appropriate `IntegerVectorProperty` for each kind of shape (supporting also straight-up inf), which is declared to the node that supports array-likeness so it can decide how exactly to expose properties in the array-like context of things.
[ ] Make `to_socket`s no-consent to new links from `from_socket`s of differing type (we'll see if this controls the typing story enough for now, and how much we'll need capabilities in the long run)
- [?] Alternatively, reject non matching link types, and red-mark non matching capabilities?

## Many Nodes
[ ] Implement LazyValue stuff, including LazyParamValue on a new class of constant-like input nodes that really just emit ex. sympy variables.
[?] Require a Unit System for nodes that construct Tidy3D objects
[ ] Medium Features
- [ ] Accept spatial field. Else, spatial uniformity.
- [ ] Accept non-linearity. Else, linear.
- [ ] Accept space-time modulation. Else, static.
[ ] Modal Features
- [ ] ModeSpec, for use by ModeSource, ModeMonitor, ModeSolverMonitor. Data includes ModeSolverData, ModeData, ScalarModeFieldDataArray, ModeAmpsDataArray, ModeIndexDataArray, ModeSolver.

## Development Tooling
[ ] Implement `rye` support
[ ] Setup neovim to be an ideal editor

## Version Churn
[ ] Implement real StrEnum sockets, since they appear in py3.11
[ ] Think about implementing new panels where appropriate (<https://developer.blender.org/docs/release_notes/4.1/python_api/>)
[ ] Think about using the new bl4.1 file handler API to enable drag and drop creation of appropriate nodes (for importing files without hassle).
[ ] Keep an eye on our manual `__annotations__` hacking; python 3.13 is apparently fucking with it.
[ ] Plan for multi-input sockets <https://projects.blender.org/blender/blender/commit/14106150797a6ce35e006ffde18e78ea7ae67598> (for now, just use the "Combine" node and have seperate socket types for both).
[ ] Keep an eye out for volume geonodes in 4.2 (July 16, 2024), which will better allow for more complicated volume processing (we might still want/need the jax based stuff after, but let's keep it minimal just in case)

## Packaging
[ ] Allow specifying custom dir for keeping pip dependencies, so we can unify prod and dev (currently we hard-code a dev dependency path).
[ ] Refactor top-level `__init__.py` to check dependencies first. If not everything is available, it should only register a minimal addon; specifically, a message telling the user that the addon requires additional dependencies (list which), and the button to install them. When the installation is done, re-check deps and register the rest of the addon.
[ ] Use a Modal and multiline-text-like construction to print `pip install` as we install dependencies, so that the user has an idea that something is happening.
[ ] Test on Windows

## Node Tree Cache Semantics
