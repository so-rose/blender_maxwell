# Acute Tasks
- [x] Implement Material Import for Maxim Data
- [x] Implement Robust DataFlowKind for list-like / spectral-like composite types
- [ ] Unify random node/socket caches.
- [ ] Finish the "Low-Hanging Fruit" Nodes
- [ ] Move preview GN trees to the asset library.



# Nodes
## Analysis
- [ ] Extract
- [ ] Viz

## Inputs
- [x] Wave Constant
	- [x] Implement export of frequency / wavelength array/range.
- [x] Unit System
	- [ ] Implement presets, including "Tidy3D" and "Blender", shown in the label row.

- [x] Constants / Scientific Constant
	- [x] Create `utils.sci_constants` to map `scipy` constants to `sympy` units.
	- [x] Utilize `utils.sci_constants` to make it easy for the user to select appropriate constants with two-layered dropdowns.
- [x] Constants / Number Constant
- [ ] Constants / Physical Constant
	- [ ] Pol: Elliptical viz as 2D plot.
	- [ ] Pol: Poincare sphere viz as 3D GN.
- [x] Constants / Blender Constant

- [ ] Web / Tidy3D Web Importer
	- [ ] Change to output only a `FilePath`, which can be plugged into a Tidy3D File Importer.
	- [ ] Implement caching, such that the file will only download if the file doesn't already exist.
	- [ ] Have a visual indicator for the current download status, with a manual re-download button.

- [x] File Import / Material Import
	- [x] Dropdown to choose import format
	- MERGED w/TIDY3D FILE IMPORT
- [x] File Import / Tidy3D File Import
	- [x] HDF and JSON file support, with appropriate choice of loose output socket.
- [ ] File Import / Array File Import
	- [ ] Standardize 1D and 2D array loading/saving on numpy's savetxt with gzip enabled.
	- [ ] Implement unit system input to guide conversion from numpy data type.
	- [ ] Implement datatype dropdown to guide format from disk, prefilled to detected.
	- [ ] Implement a LazyValue to provide a data path that avoids having to load massive arrays every time always.

## Outputs
- [x] Viewer
	- [ ] Remove image preview when disabling plots.
	- [x] Auto-enable 3D preview when creating.
	- [ ] Test/support multiple viewers at the same time.
	- [ ] Pop-up w/multiline string as alternative to console print.

- [x] Web Export / Tidy3D Web Exporter
	- This is an extraordinarily nuanced node, and will need constant adjusting if it is to be robust.
	- [ ] We need better ways of doing checks before uploading, like for monitor data size. Maybe a SimInfo node?
	- [ ] Implement "new folder" feature w/popup operator.
	- [ ] Implement "delete task" feature w/popup confirmation.
	- [ ] We need to be able to "delete and re-upload" (or maybe just delete from the interface).

- [x] File Export / JSON File Export
- [ ] File Import / Tidy3D File Export
	- [ ] Implement HDF-based export of Tidy3D-exported object (which includes ex. mesh data and such)
	- [ ] Also JSON (but indicate somehow that ex. mesh data doesn't come along for the ride).
- [ ] File Export / Array File Export
	- [ ] Implement datatype dropdown to guide format on disk.
	- [ ] Implement unit system input to guide conversion to numpy data type.
	- [ ] Standardize 1D and 2D array loading/saving on numpy's savetxt with gzip enabled.

## Viz
- [x] Monitor Data Viz
	- [x] Implement dropdown to choose which monitor in the SimulationData should be visualized (based on which are available in the SimulationData), and implement visualization based on every kind of monitor-adjascent output data type (<https://docs.flexcompute.com/projects/tidy3d/en/latest/api/output_data.html>)
	- [ ] Project field values onto a plane object (managed)

## Sources
- [x] Temporal Shapes / Gaussian Pulse Temporal Shape
- [x] Temporal Shapes / Continuous Wave Temporal Shape
- [ ] Temporal Shapes / Symbolic Temporal Shape
	- [ ] Specify a Sympy function to generate appropriate array based on
- [ ] Temporal Shapes / Array Temporal Shape

- [x] Point Dipole Source
	- [ ] Use a viz mesh, not empty (empty doesn't play well with alpha hashing).
- [ ] Plane Wave Source
	- [x] Implement an oriented vector input with 3D preview.
	- [ ] **IMPORTANT**: Fix the math so that an actually valid construction emerges!!
- [ ] Uniform Current Source
- [ ] TFSF Source

- [ ] Gaussian Beam Source
- [ ] Astigmatic Gaussian Beam Source

- [ ] Mode Source

- [ ] Array Source / EH Array Source
- [ ] Array Source / EH Equiv Array Source

## Mediums
- [x] Library Medium
	- [ ] Implement frequency range output (listy)
- [ ] PEC Medium
- [ ] Isotropic Medium
- [ ] Anisotropic Medium

- [ ] Sellmeier Medium
- [ ] Drude Medium
- [ ] Drude-Lorentz Medium
- [ ] Debye Medium
- [ ] Pole-Residue Medium
	
- [ ] Non-Linearity / `chi_3` Susceptibility Non-Linearity
- [ ] Non-Linearity / Two-Photon Absorption Non-Linearity
- [ ] Non-Linearity / Kerr Non-Linearity

- [ ] Space/Time epsilon/mu Modulation

## Structures
- [ ] BLObject Structure
- [x] GeoNodes Structure
	- [x] Rewrite the `bl_socket_map.py`
	- [x] Use the modifier itself as memory, via the ManagedObj
	- [x] Rewrite to use unit systems properly.
	- [ ] Propertly map / implement Enum input sockets to the GN group.
	- [ ] Implement a panel system, either based on native GN panels, or description parsing, or something like that.

- [ ] Primitive Structures / Plane Structure
- [x] Primitive Structures / Box Structure
- [x] Primitive Structures / Sphere Structure
- [ ] Primitive Structures / Cylinder Structure
- [ ] Primitive Structures / Ring Structure
- [ ] Primitive Structures / Capsule Structure
- [ ] Primitive Structures / Cone Structure

## Monitors
- [x] E/H Field Monitor
- [x] Field Power Flux Monitor
- [ ] \epsilon Tensor Monitor
- [ ] Diffraction Monitor
- [ ] Axis-aligned planar 2D (pixel)

- [ ] Projected E/H Field Monitor / Cartesian Projected E/H Field Monitor
	- [ ] Use to implement the metalens: <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/Metalens.html>
- [ ] Projected E/H Field Monitor / Angle Projected E/H Field Monitor
- [ ] Projected E/H Field Monitor / K-Space Projected E/H Field Monitor

- [ ] Modal Nodes
	- Spatial+frequency feature monitoring. An EM field can be decomposed into using a specially configured solver, which can be used to look for very particular kinds of effects by constraining investigations of a solver result to filter out everything that isn't these particular modes aka. features. Kind of a fourier-based redimensionalization, almost).

## Simulations
- [x] FDTDSim

- [x] Sim Domain
	- [ ] By-Medium batching of Structures when building the td.Simulation object, which can have significant performance implications.

- [x] Boundary Conds
- [ ] Boundary Cond / PML Bound Face
	- [ ] Dropdown for "Normal" and "Stable"
- [ ] Boundary Cond / PEC Bound Face
- [ ] Boundary Cond / PMC Bound Face
- [ ] Boundary Cond / Bloch Bound Face
- [ ] Boundary Cond / Periodic Bound Face
- [ ] Boundary Cond / Absorbing Bound Face

- [ ] Sim Grid
- [ ] Sim Grid Axes / Auto Sim Grid Axis
- [ ] Sim Grid Axes / Manual Sim Grid Axis
- [ ] Sim Grid Axes / Uniform Sim Grid Axis
- [ ] Sim Grid Axes / Array Sim Grid Axis

## Utilities
- [ ] Separate
- [x] Combine
	- [x] Implement concatenation of sim-critical socket types into their multi-type



# GeoNodes
- [ ] Tests / Monkey (suzanne deserves to be simulated, she may need manifolding up though :))
- [ ] Tests / Wood Pile

- [ ] Structures / Primitives / Plane
- [x] Structures / Primitives / Box
- [x] Structures / Primitives / Sphere
- [ ] Structures / Primitives / Cylinder
- [x] Structures / Primitives / Ring
- [ ] Structures / Primitives / Capsule
- [ ] Structures / Primitives / Cone

- [ ] Structures / Arrays / Square
- [ ] Structures / Arrays / Square-Hole
- [ ] Structures / Arrays / Cyl
- [ ] Structures / Arrays / Cyl-Hole
- [x] Structures / Arrays / Box
- [x] Structures / Arrays / Sphere
- [ ] Structures / Arrays / Cylinder
- [x] Structures / Arrays / Ring
- [ ] Structures / Arrays / Capsule
- [ ] Structures / Arrays / Cone

- [ ] Array / Square Array **NOTE: Ring and cylinder**
- [ ] Array / Hex Array **NOTE: Ring and cylinder**
- [ ] Hole Array / Square Hole Array: Takes a primitive hole shape.
- [ ] Hole Array / Hex Hole Array: Takes a primitive hole shape.
- [ ] Cavity Array / Hex Array w/ L-Cavity
- [ ] Cavity Array / Hex Array w/ H-Cavity

- [ ] Crystal Sphere Lattice / Sphere FCC Array
- [ ] Crystal Sphere Lattice / Sphere BCC Array



# Benchmark / Example Sims
- [ ] Research-Grade Experiment
	- Membrane 15nm thickness suspended in air
	- Square lattice of holes period 900nm (900nm between each hole, air inside holes)
	- Holes square radius 100nm
	- Square lattice
	- Analysis of transmission
	- Guided mode resonance
- [ ] Tunable Chiral Metasurface <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/TunableChiralMetasurface.html>



# Sockets
## Basic
- [x] Any
- [x] Bool
- [x] String
- [x] File Path
- [x] Color

## Number
- [x] Integer
- [x] Rational
- [x] Real
	- [ ] Implement min/max for ex. 0..1 factor support.
- [x] Complex

## Blender
- [x] Object
	- [ ] Implement default object name in SocketDef
- [x] Collection
	- [ ] Implement default collection name in SocketDef 

- [x] Image
	- [ ] Implement default image name in SocketDef

- [x] GeoNodes
	- [ ] Implement default SocketDef geonodes name
- [x] Text
	- [ ] Implement default SocketDef object name

## Maxwell
- [x] Bound Conds
- [ ] Bound Cond

- [x] Medium
- [ ] Medium Non-Linearity

- [x] Source
- [ ] Temporal Shape
	- [ ] Sane-default pulses for easy access.

- [ ] Structure
- [ ] Monitor

- [ ] FDTD Sim
- [ ] Sim Domain
	- [ ] Toggleable option to push-sync the simulation time duration to the scene end time (how to handle FPS vs time-step? Should we adjust the FPS such that there is one time step per frame, while keeping the definition of "second" aligned to the Blender unit system?)
- [ ] Sim Grid
- [ ] Sim Grid Axis

- [ ] Simulation Data

## Tidy3D
- [x] Cloud Task
	- [ ] Implement switcher for API-key-having config filconfig file vs. direct entry of API key. It should be auto-filled with the config file when such a thing exists.

## Physical
- [x] Unit System
	- [ ] Presets for Blender and Tidy3D
	- [ ] Dropdowns in the socket UI

- [x] Time

- [x] Angle
	- [ ] Remove superfluous units.
- [ ] Solid Angle (steradian)

- [x] Frequency (hertz)
- [ ] Angular Frequency (`rad*hertz`)
### Cartesian
- [x] Length
- [x] Area
- [x] Volume

- [ ] Point 1D
- [ ] Point 2D
- [x] Point 3D

- [ ] Size 2D
- [x] Size 3D

- [ ] Rotation 3D
	- [ ] Implement Euler methods
	- [ ] Implement Quaternion methods
### Mechanical
- [ ] Mass

- [x] Speed
- [ ] Velocity 3D
- [x] Acceleration Scalar
- [ ] Acceleration 3D
- [x] Force Scalar
- [ ] Force 3D
- [ ] Pressure
### Energy
- [ ] Energy (joule)
- [ ] Power (watt)
- [ ] Temperature
### Electrodynamical
- [ ] Current (ampere)
- [ ] Current Density 3D

- [ ] Charge (coulomb)
- [ ] Voltage (volts)
- [ ] Capacitance (farad)
- [ ] Resistance (ohm)
- [ ] Electric Conductance (siemens)

- [ ] Magnetic Flux (weber)
- [ ] Magnetic Flux Density (tesla)
- [ ] Inductance (henry)

- [ ] Electric Field 3D (`volt*meter`)
- [ ] Magnetic Field 3D (tesla)
### Luminal
- [ ] Luminous Intensity (candela)
- [ ] Luminous Flux (lumen)
- [ ] Illuminance (lux)
### Optical
- [ ] Jones Polarization
- [ ] Polarization (Stokes)



# Internal / Architecture
## CRITICAL
- [ ] License header UI for MaxwellSimTrees, to clarify the AGPL-compatible potentially user-selected license that trees must be distributed under.
- [ ] Document the node tree cache semantics thoroughly; it's a VERY nuanced piece of logic, and its invariants may not survive Blender versions / the author's working memory
- [ ] Start standardizing nodes/sockets w/individualized SemVer
	- Perhaps keep node / socket versions in a property, so that trying to load an incompatible major version hop can error w/indicator of where to find a compatible `blender_maxwell` version.
- [ ] `log.error` should invoke `self.report` in some Blender operator - used for errors that are due to usage error (which can't simply be prevented with UX design, like text file formatting of import), not due to error in the program.

## Documentation
- [ ] Make all modules available
- [ ] Publish documentation site.
- [ ] Initial user guides w/pictures.
- [ ] Comb through and finish `__doc__`s.

## Performance
- [ ] The GN value pushing currently does an expensive from-sympy conversion for all GN attributes, making it very slow.
	- Generally, the issue is that we can't compare the pushed value to the existing value without a conversion.
	- Also, sympy expressions can't be hashed (by default), and `str()` may be just as slow, so a simple `@lru_cache` is no good.
	- **One approach** is a `HashableSympyExpr` class, which would use ex. atoms and values in `SympyExpr` to let us `@lru_cache` the sympy conversion.
	- **Another approach** is to "just" make the `scale_to_unit` function faster (we should profile).
		- Presumably, the scaling factors between units could be eagerly cached, then the numerical part of the expression could be used to avoid `convert_to` calls.
		- Without the list format of `(np.array, spu.Quantity)`, `sp.Matrix` can't be performantly handled like this.

## Style
- [ ] Color of nodes should be a nice blue, eh?

## Registration and Contracts
- [ ] Refactor the node category code; it's ugly.
	- It's maybe not that easy. And it seems to work with surprising reliability. Leave it alone for now!
- [ ] (?) Would be nice with some kind of indicator somewhere to help set good socket descriptions when making geonodes.

## Managed Objects
- [ ] Implement ManagedEmpty
	- [ ] Implement image-based empty connected to an image (which is managed by a different ManagedImage owned by the same node instance)
- [ ] Implement ManagedVol
	- [ ] Implement loading the xarray-defined voxels into OpenVDB, saving it, and loading it as a managed BL object with the volume setting.
	- [ ] Implement basic jax-driven volume voxel processing, especially cube based slicing.
	- [ ] Implement jax-driven linear interpolation of volume voxels to an image texture, whose pixels are sized according to the dimensions of another managed plane object (perhaps a uniquely described Managed BL object itself).

## Utils or Services
- [ ] Document the `tdcloud` service thoroughly and open a GitHub discussion about `td.web` shortcomings.

## Node Base Class
- [ ] Implement a new socket type for preview-only parameters
	- [ ] When used, the node base class should expose a toggle 
	- [ ] Instead of mangling props, we can instead reuse all of the socket-based code, while also enabling composability of previews.
- [ ] Custom `@cache`/`@lru_cache`/`@cached_property` which caches by instance ID (possibly based on `beartype` or `pydantic`).
	- The problem to solve is performance vs. persistence.
- [ ] 
- [ ] Implement by-category sorting of loose sockets with 'move' method on `node.inputs`/`node.outputs`.
	- Currently order is not guaranteed
- [ ] When presets are used, if a preset is selected and the user alters a preset setting, then dynamically switch the preset indicator back to "Custom"  to indicate that there is no active preset

## Events
- [x] Mechanism for selecting a blender object managed by a particular node.
	- [ ] Standard way of triggering the selection
- [ ] Mechanism for ex. specially coloring a node that is currently participating in the preview.
- [ ] Custom callbacks when deleting a node (in `free()`), to ex. delete all previews with the viewer node.

## Socket Base Class
- [ ] Second-generation listy, based on a `DataFlowKind.ValueListy`, `DataFlowKind.ValueRange`, etc. to encode the presence of special logic.
	- This is key to allow special handling, as "just give me a `list[]` of `sympy` objects" is an exceptionally non-performant and brittle thing.
- [ ] Implement capability system, which defaults to exactly matching the type.
	- [ ] Make `to_socket`s no-consent to new links from `from_socket`s of incompatible Capability.
	- [ ] Add Capabilities needed mainly in cases where we need `Any` currently.

## Many Nodes
- [ ] Implement "Steady-State" / "Time Domain" on all relevant Monitor nodes
- [ ] (?) Dynamic `bl_label` where appropriate (ex. "Library Medium" becoming "Au Medium")
- [ ] Implement LazyValue, including LazyParamValue on a new class of constant-like input nodes that really just emit ex. sympy variables.
- [ ] Medium Features
	- [ ] Accept spatial field. Else, spatial uniformity.
	- [ ] Accept non-linearity. Else, linear.
	- [ ] Accept space-time modulation. Else, static.
- [ ] Modal Features
	- [ ] ModeSpec, for use by ModeSource, ModeMonitor, ModeSolverMonitor. Data includes ModeSolverData, ModeData, ScalarModeFieldDataArray, ModeAmpsDataArray, ModeIndexDataArray, ModeSolver.

## Many Sockets
- [ ] Implement constrained SympyExpr checks all over the place.

## Development Tooling
- [ ] Pass a `mypy` check
- [ ] Pass all `ruff` checks, including `__doc__` availability.
- [ ] Implement `pre-commit.
- [ ] Add profiling support, so we can properly analyze performance characteristics.
	- Without a test harness, or profile-while-logging, there may be undue noise in our analysis.
- [ ] Simple `pytest` harnesses for unit testing of nodes, sockets.
	- Start with the low-hanging-fruit stuff. Eventually, work towards wider code coverage w/headless Blender.

## Version Churn
- [ ] Migrate to StrEnum sockets (py3.11).
- [ ] Implement drag-and-drop node-from-file via bl4.1 file handler API.
- [ ] Start thinking about ways around `__annotations__` hacking.
- [ ] Prepare for for multi-input sockets (bl4.2)
	- PR has been merged: <https://projects.blender.org/blender/blender/commit/14106150797a6ce35e006ffde18e78ea7ae67598> (for now, just use the "Combine" node and have seperate socket types for both).
	- The `Combine` node has its own benefits, including previewability of "only structures". Multi-input would mainly be a kind of shorthand in specific cases (like input to the `Combine` node?)
- [ ] Prepare for volume geonodes (bl4.2; July 16, 2024)
	- Will allow for actual volume processing in GeoNodes.
	- We might still want/need the jax based stuff after; volume geonodes aren't finalized.

## Packaging
- [ ] Popup to install dependencies after UI is available (possibly with the help of the `draw()` function of the `InstallPyDeps` operator)
- [ ] Use a Modal and multiline-text-like construction to print `pip install` as we install dependencies, so that the user has an idea that something is happening.
- [ ] Test lockfile platform-agnosticism on Windows





# BUGS
We're trying to do our part by reporting bugs we find!
This is where we keep track of them for now.

## Blender Maxwell Bugs
- [ ] BUG: CTRL+SHIFT+CLICK not on a node shows an error; should just do nothing.
- [ ] Slow changing of socket sets / range on wave constant.
- [ ] API auth shouldn't show if everything is fine in Cloud Task socket
- [ ] Cloud task socket loads folders before its node shows, which can be slow (and error prone if offline)
- [ ] Dispersive fit is slow, which means lag on normal operations that rely on the fit result - fit computation should be integrated into the node, and the output socket should only appear when the fit is available.
- [ ] Numerical, Physical Constant is missing entries

BROKE NODES
- [ ] Numerical constant doesn't switch types
- [ ] Blender constant is inexplicably mega laggy
- [ ] Web importer is just wonky in general
- [ ] JSON File exporter is having trouble with generic types (is that bad?)

- [ ] Extact Data needs flux settings
- [ ] Point dipole still has no preview
- [ ] Plane wave math still doesn't work and it has no preview
- [ ] Monitors need a way of setting infinite dimensions

## Blender Bugs
Reported:
- (SOLVED) <https://projects.blender.org/blender/blender/issues/119664>

Unreported:
- The `__mp_main__` bug.
- Animated properties within custom node trees don't update with the frame. See: <https://projects.blender.org/blender/blender/issues/66392>

## Tidy3D bugs
Unreported:
- Directly running `SimulationTask.get()` is missing fields - it doesn't return some fields, including `created_at`. Listing tasks by folder is not broken.
