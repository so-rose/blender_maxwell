# Working TODO
- [x] Wave Constant
- Bounds
	- [ ] Boundary Conds
	- [ ] PML
	- [ ] PEC
	- [ ] PMC
	- [ ] Bloch
	- [ ] Absorbing
- Sources
	- [ ] Temporal Shapes / Continuous Wave Temporal Shape
	- [ ] Temporal Shapes / Symbolic Temporal Shape
	- [ ] Plane Wave Source
	- [ ] TFSF Source
	- [ ] Gaussian Beam Source
	- [ ] Astig. Gauss Beam
- Material Data Fitting
	- [ ] Data File Import
	- [ ] DataFit Medium
- Monitors
	- [x] EH Field
	- [x] Power Flux
	- [ ] Permittivity
	- [ ] Diffraction
- Structures
	- [ ] Cylinder
	- [ ] Cylinder Array
	- [ ] L-Cavity Cylinder
	- [ ] H-Cavity Cylinder
	- [ ] FCC Lattice
	- [ ] BCC Lattice
	- [ ] Monkey
- Sim Grid
	- [ ] Sim Grid
	- [ ] Auto
	- [ ] Manual
	- [ ] Uniform
	- [ ] Data
- Mediums
	- [ ] PEC Medium
	- [ ] Isotropic Medium
	- [ ] Sellmeier Medium
	- [ ] Drude Medium
	- [ ] Debye Medium
	- [ ] Anisotropic Medium
- Tidy3D
	- [ ] Exporter
	- [ ] Importer
- Integration
	- [ ] Simulation and Analysis of Maxim's Cavity
- Constants
	- [x] Number Constant
	- [x] Vector Constant
	- [x] Physical Constant

- [ ] Fix many problems by persisting `_enum_cb_cache` and `_str_cb_cache`.




# Nodes
## Analysis
- [x] Extract
	- [ ] Implement "saved" state for info, and provide the user an indicator that state has been saved w/a button to reset (the state should also be reset when plugging a new data thing in)
- [x] Viz
	- [ ] Implement Info-driven planar projection of pixels onto managed image empty.
	- [ ] Live-slice 2D field values onto user-controlled image empty from 2D field.
	- [ ] SocketType-based visualization support.
		- [ ] Pol SocketType: 2D elliptical visualization of Jones vectors.
		- [ ] Pol SocketType: 3D Poincare sphere visualization of Stokes vectors.

- [x] Math / Map Math
	- [x] Remove "By x" socket set let socket sets only be "Function"/"Expr"; then add a dynamic enum underneath to select "By x" based on data support.
	- [ ] Filter the operations based on data support, ex. use positive-definiteness to guide cholesky.
	- [ ] Implement support for additional symbols via `Expr`.
- [x] Math / Filter Math
- [ ] Math / Reduce Math
- [x] Math / Operate Math
	- [ ] Remove two-layered dropdown; directly filter operations and use categories to seperate them.
	- [ ] Implement Expr socket advancements to make a better experience operating between random expression-like sockets.

## Inputs
- [x] Wave Constant
- [x] Scene
	- [ ] Implement export of scene time via. Blender unit system.
	- [ ] Implement optional scene-synced time exporting, so that the simulation definition and scene definition match for analysis needs.

- [x] Constants / Expr Constant
	- See IDEAS.
- [x] Constants / Number Constant
- [x] Constants / Vector Constant
- [x] Constants / Physical Constant
- [x] Constants / Scientific Constant
	- [ ] Nicer (boxed?) node information, maybe centered headers, in a box, etc. .
- [ ] Constants / Unit System Constant
	- [ ] Re-implement with `PhysicalType`.
	- [ ] Implement presets, including "Tidy3D" and "Blender", shown in the label row.
- [ ] Constants / Blender Constant
	- [ ] Fix it!

- [ ] Web / Tidy3D Web Importer
	- [ ] Fix the check of folders, actually, just fix `tdcloud` in general!
	- [ ] Have a visual indicator for the download status of the currently selected task, as well as its data size.
	- [ ] If a task is "selected", lock the cloud task socket, so other tasks can't be selected. While that lock is active, expose a real "download" button. Also make the loose output socket and put out a `FlowPending` until the download is available.
	- [ ] A manual download button and seperate re-download button (maybe on the side, round reload boi).
	- [ ] An option to pack the data into the blend, with overview of how much data it will take (Base85/base64 has overhead).
	- [ ] Default limits for caching/packing.
	- [ ] Support importing batched simulations and outputting an `Array` of SimData.

- [ ] File Import / Data File Import
	- [ ] Implement `FlowKind.LazyValueFunc` that plays the loading game.
	- [ ] Implement `FlowKind.Info` which lets the user describe the data being loaded, for proper further processing.
		- [ ] Implement unit system input to guide conversion from numpy data type.
		- [ ] Implement datatype dropdown to guide format from disk, prefilled to detected.
	- [ ] Implement `FlowKind.Array` that just runs the `LazyValueFunc` as usual.
	- [ ] Standardize 1D and 2D array loading/saving on numpy's savetxt with gzip enabled.
- [x] File Import / Tidy3D File Import

## Outputs
- [x] Viewer
	- [ ] Consider a "debug" mode
	- [ ] Auto-enable plot when creating.
	- [ ] Test/support multiple viewers at the same time.
	- [ ] Pop-up w/multiline string as alternative to console print.
	- [ ] Handle per-tree viewers, so that switching trees doesn't "bleed" state from the old tree.
	- [ ] BUG: CTRL+SHIFT+CLICK not on a node shows an error; should just do nothing.

- [x] Web Export / Tidy3D Web Exporter
	- [ ] Run checks on-demand, and require they be run before the sim can be uploaded. If the simulation changes, don't
	- [ ] Support doing checks in a seperate process.
	- [ ] We need better ways of doing checks before uploading, like for monitor data size. Maybe a SimInfo node?
	- [ ] Accept `Array` of simulations, and upload them as `Batch`.

- [x] File Export / JSON File Export
	- [ ] Reevaluate its purpose.
- [ ] File Export / Tidy3D File Export
	- [ ] Implement HDF-based export of Tidy3D-exported object (which includes ex. mesh data and such)
	- [ ] Also JSON (but indicate somehow that ex. mesh data doesn't come along for the ride).
- [ ] File Export / Data File Export
	- [ ] Implement datatype dropdown to guide format on disk.
	- [ ] Implement unit system input to guide conversion to numpy data type.
	- [ ] Standardize 1D and 2D array loading/saving on numpy's savetxt with gzip enabled.

## Sources
- [x] Temporal Shapes / Gaussian Pulse Temporal Shape
- [x] Temporal Shapes / Continuous Wave Temporal Shape
- [ ] Temporal Shapes / Symbolic Temporal Shape
	- [ ] Specify a Sympy function to generate appropriate array based on
- [ ] Temporal Shapes / Data Temporal Shape

- [x] Point Dipole Source
	- [ ] Use a viz mesh, not empty (empty doesn't play well with alpha hashing).
- [ ] Plane Wave Source
	- [ ] **IMPORTANT**: Fix the math so that an actually valid construction emerges!!
- [ ] Uniform Current Source
- [ ] TFSF Source

- [ ] Gaussian Beam Source
- [ ] Astigmatic Gaussian Beam Source

- [ ] EH Array Source
- [ ] EH Equiv Array Source

## Mediums
- [x] Library Medium
	- [ ] Implement frequency range output (listy), perhaps in the `InfoFlow` lane?
	- [ ] Implement dynamic label.
- [ ] DataFit Medium
	- [ ] Implement by migrating the material data fitting logic from the `Tidy3D File Importer`, except now only accept a `Data` input socket, and rely on the `Data File Importer` to do the parsing into an acceptable `Data` socket format.
	- [ ] Save the result in the node, specifically in a property (serialized!) and lock the input graph while saved.

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

## Structures
- [ ] BLObject Structure
- [x] GeoNodes Structure
	- [ ] Implement a panel system, to make GN trees with a ton of inputs (most of which are not usually needed) actually useful.
	- [ ] Propertly map / implement Enum input sockets to the GN group.

- [ ] Primitive Structures / Line Structure
- [ ] Primitive Structures / Plane Structure
- [x] Primitive Structures / Box Structure
- [x] Primitive Structures / Sphere Structure
- [ ] Primitive Structures / Cylinder Structure
- [ ] Primitive Structures / PolySlab Structure

## Bounds
- [x] Boundary Conds
- [ ] Boundary Cond / PML Bound Face
	- [ ] Dropdown for "Normal" and "Stable"
- [ ] Boundary Cond / PEC Bound Face
- [ ] Boundary Cond / PMC Bound Face
- [ ] Boundary Cond / Bloch Bound Face
	- [ ] Implement "simple" mode aka "periodic" mode in Tidy3D
- [ ] Boundary Cond / Absorbing Bound Face

## Monitors
- [x] EH Field Monitor
	- [ ] Method of setting `inf` on dimensions - use a `ManyEnum` maybe to select the injection axis, and let that set the $0$.
	- [ ] Revamp the input parameters.
- [x] Power Flux Monitor
- [ ] Permittivity Monitor
- [ ] Diffraction Monitor

- [ ] Projected E/H Field Monitor / Cartesian Projected E/H Field Monitor
	- [ ] Use to implement the metalens: <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/Metalens.html>
- [ ] Projected E/H Field Monitor / Angle Projected E/H Field Monitor
- [ ] Projected E/H Field Monitor / K-Space Projected E/H Field Monitor

## Simulations
- [x] FDTDSim
	- [ ] By-Medium batching of Structures when building the td.Simulation object, which can have significant performance implications.

- [x] Sim Domain

- [ ] Sim Grid
- [ ] Sim Grid Axes / Auto Sim Grid Axis
- [ ] Sim Grid Axes / Manual Sim Grid Axis
- [ ] Sim Grid Axes / Uniform Sim Grid Axis
- [ ] Sim Grid Axes / Data Sim Grid Axis

## Utilities
- [ ] Separate
	- [ ] Use generic Expr socket mode to combine numerical types into either Expr or Data socket.
- [x] Combine
	- [ ] Use generic Expr socket mode to combine numerical types into either Expr or Data socket.
	- [ ] Explicit about lower structures taking precedence.



# GeoNodes
- [ ] Tests / Monkey (suzanne deserves to be simulated, she may need manifolding up though :))
- [ ] Tests / Wood Pile

- [ ] Structures / Primitives / Line
- [ ] Structures / Primitives / Plane
- [x] Structures / Primitives / Box
- [x] Structures / Primitives / Sphere
- [ ] Structures / Primitives / Cylinder
- [x] Structures / Primitives / Ring

- [ ] Structures / Arrays / Cyl
- [ ] Structures / Arrays / Box
- [ ] Structures / Arrays / Sphere
- [ ] Structures / Arrays / Cylinder
- [x] Structures / Arrays / Ring

- [ ] Structures / Hex Arrays / Cyl
- [ ] Structures / Hex Arrays / Box
- [ ] Structures / Hex Arrays / Sphere
- [ ] Structures / Hex Arrays / Cylinder
- [x] Structures / Hex Arrays / Ring

- [ ] Structures / Cavity Arrays / L-Cavity Cylinder
- [ ] Structures / Cavity Arrays / H-Cavity Cylinder

- [ ] Structures / Lattice Arrays / FCC Sphere
- [ ] Structures / Lattice Arrays / BCC Sphere



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
- [x] Expr
	- [ ] Implement node-driven support for dynamic symbols.
	- [ ] Implement compatibility with sockets that fundamentally do produce expressions, especially Physical sockets.

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
	- [ ] Move API checking out of the socket, and don't re-prompt for a key if the config file exists.
	- [ ] Remove the existing task selector when making a new task.
	- [ ] Implement "new folder" feature w/popup operator.
	- [ ] Implement "delete task" feature w/popup confirmation.

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
- [ ] Rethink the way that loose sockets are replaced, specifically with respect to deterministic ordering.
	- Currently order is not guaranteed. This is causing problems.


## User-Facing Errors and Legal Considerations
- [ ] `log.error` should invoke `self.report` in some Blender operator - used for errors that are due to usage error (which can't simply be prevented with UX design, like text file formatting of import), not due to error in the program.
- [ ] License header UI for MaxwellSimTrees, to clarify the AGPL-compatible potentially user-selected license that trees must be distributed under.
- [ ] A "CitationsFlow" FlowKind which simply propagates citations.
- [ ] Implement standardization of nodes/sockets w/individualized SemVer
	- Perhaps keep node / socket versions in a property, so that trying to load an incompatible major version hop can error w/indicator of where to find a compatible `blender_maxwell` version.
	- Integrate w/BLField, to help the user manage addon updates that would break their tree.

## Documentation
- [ ] Make all modules available
- [ ] Publish documentation site.
- [ ] Initial user guides w/pictures.
- [ ] Comb through and finish `__doc__`s.

## Performance
- [ ] Optimize GN value pushing w/sympy expression hashing.

## Style
Header color style can't be done, unfortunately. Body color feels unclean, so nothing there for now.

- [ ] Node icons to denote preview/plot state.

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
- [ ] Re-engineer "presets" to use an Enum of some kind.

## Events
- [ ] When a Blender object is selected, select the node that owns its ManagedObj.
- [ ] Node button / shortcut / something to select the ManagedObj owned by a node.
- Sync transformation of Blender object by user to its node properties.
	- See <https://archive.blender.org/developer/P563>
	- Also see <https://blender.stackexchange.com/questions/150809/how-to-get-an-event-when-an-object-is-selected>

## Socket Base Class
- [ ] Collect `SocketDef` objects like we do with `BL_REGISTER`, without any special mojo sauce.

## Many Nodes
- [ ] Implement "Steady-State" / "Time Domain" on all relevant Monitor nodes
- [ ] Medium Features
	- [ ] Accept spatial field. Else, spatial uniformity.
	- [ ] Accept non-linearity. Else, linear.
	- [ ] Accept space-time modulation. Else, static.
- [ ] Modal Features
	- ModeSpec, for use by ModeSource, ModeMonitor, ModeSolverMonitor. Data includes ModeSolverData, ModeData, ScalarModeFieldDataArray, ModeAmpsDataArray, ModeIndexDataArray, ModeSolver.

## Many Sockets

## Development Tooling
- [ ] Implement `pre-commit.
- [ ] Pass a `mypy` check
- [ ] Pass all `ruff` checks, including `__doc__` availability.
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
- [ ] Test lockfile platform-agnosticism on Windows

## BLCache
- [ ] Replace every raw property with `BLField`.
- [ ] Add matrix property support: https://developer.blender.org/docs/release_notes/3.0/python_api/#other-additions
- [ ] Fix many problems by persisting `_enum_cb_cache` and `_str_cb_cache`.
- [ ] Docstring parser for descriptions.
- [ ] Method of dynamically setting property options after creation, using `idproperty_ui_data`





# BUGS
We're trying to do our part by reporting bugs we find!
This is where we keep track of them for now, if they're not covered by the above listings.

## Blender Maxwell Bugs

## Blender Bugs
Reported:
- (SOLVED) <https://projects.blender.org/blender/blender/issues/119664>

Unreported:
- Units are unruly, and are entirely useless when it comes to going small like this.
- The `__mp_main__` bug.
- Animated properties within custom node trees don't update with the frame. See: <https://projects.blender.org/blender/blender/issues/66392>
- Can't update `items` using `id_properties_ui` of `EnumProperty`. Maybe less a bug than an annoyance.
- **Matrix Display Bug**: The data given to matrix properties is entirely ignored in the UI; the data is flattened, then left-to-right, up-to-down, the data is inserted. It's neither row-major nor column-major - it's completely flat.
	- Though, if one wanted row-major (**as is consistent with `mathutils.Matrix`**), one would be disappointed - the UI prints the matrix property column-major
	- Trying to set the matrix property with a `mathutils.Matrix` is even stranger - firstly, the size of the `mathutils.Matrix` must be transposed with respect to the property size (again the col/row major mismatch). But secondly, even when accounting for the col/row major mismatch, the values of a ex. 2x3 (row-major) matrix (written to with a 3x2 matrix with same flattened sequence) is written in a very strange order:
	- Write `mathutils.Matrix` `[[0,1], [2,3], [4,10]]`: Results in (UI displayed row-major) `[[0,3], [4,1], [3,5]]`
	- **Workaround (write)**: Simply flatten the 2D array, re-shape by `[cols,rows]`. The UI will display as the original array. `myarray.flatten().reshape([cols,rows])`.
	- **Workaround (read)**: `np.array([[el1 for el1 in el0] for el0 in BLENDER_OBJ.matrix_prop]).flatten().reshape([rows,cols])`. Simply flatten the property read 2D array and re-shape by `[rows,cols]`. Mind that data type out is equal to data type in.
	- Also, for bool matrices, `toggle=True` has no effect. `alignment='CENTER'` also doesn't align the checkboxes in their cells.

## Tidy3D bugs
Unreported:
- Directly running `SimulationTask.get()` is missing fields - it doesn't return some fields, including `created_at`. Listing tasks by folder is not broken.



# Designs / Proposals

## Coolness Things
- Let's have operator `poll_message_set`: https://projects.blender.org/blender/blender/commit/ebe04bd3cafaa1f88bd51eee5b3e7bef38ae69bc
- Careful, Python uses user site packages: <https://projects.blender.org/blender/blender/commit/72c012ab4a3d2a7f7f59334f4912402338c82e3c>
- Our modifier obj can see execution time: <https://projects.blender.org/blender/blender/commit/8adebaeb7c3c663ec775fda239fdfe5ddb654b06>

## IDEAS
- [ ] Depedencies-gated addon preferences.
	- [ ] Preferences-based specification/alteration of default unit systems for Tidy3D and Blender.
	- [ ] Preferences-based specification/alteration of Tidy3D API key, so we can factor away all the `prelock` bullshit.

- [ ] Subsockets
	- We need Exprs to not be so picky.
	- All the sympy-making nodes should be subsockets of Expr, so that you can plug any socket that should work with Expr into Expr.
	- When it comes to Data, any Expr that produces an array-like output from its `LazyValueFunc` should be deemed compatible (as in, the Expr may plug into a Data socket).
		- Specifically, that means the presence of a well-defined `Info`, as well as `jax` compatibility.

- [ ] Symbolic Expr Socket
	- [ ] Nodes should be able to dynamically define new symbols on their Expr sockets.
	- [ ] Expr's `FlowKind`s should be expanded:
		- [ ] `Capabilities`: Expand to include subsocket checking, where Expr is the supersocket of ex. most/all of the physical, numerical, vector sockets.
		- [ ] `Value`: Just the raw sympy expression, when `active_kind` is `Value`.
		- [ ] `Array`: The evaluated `LazyValueFunc`, when `active_kind` is `Array`.
			- Should require that the expression as a whole simplifies to `sp.Matrix`.
			- Should require that there are no symbols to be defined in a socket (since `LazyValueFunc` must be called with no args).
		- [ ] `LazyValueFunc`: Create a 'jax' function from an expression, such that each symbol becomes an argument to that function.
			- When `active_kind` is `Value`, it should take arrays/scalars and return a scalar (expression output is a normal sympy number of some kind).
			- When `active_kind` is `Array`, it should take arrays/scalars and return an array (expression output is `sp.Matrix`).
			- This kind of approach allows using 
		- [ ] `LazyValueRange`: Expose two expressions, start/end, but with one symbol set.
		- [ ] `Info`: Should always produce an `InfoFlow` that, at minimum, has an empty `dim_*`, an `output_shape` of `None`, etc., for a scalar.
	- [ ] Implement an Expr Constant node to see all this through in prototype.
		- [ ] Expr: Obviously, input and output.
		- [ ] Symbols: Node-bound dynamic thing where you can add and subtract symbols, as well as set their type. They should popup in the `Let:` statement of the input expr socket.
		- [ ] Examples: Each symbol should have the ability to set "example values", which causes the Node to fill `Params`. When all 

- [ ] Report reason for no-link using `self.report`.

- [ ] Dropping a link on empty space should query a menu of possible nodes, or if only one node is reasonable, make that node.

- [ ] Shader visualizations approximated from medium `nk` into a shader node graph, aka. a generic BSDF.

- [ ] Web importer that gets material data from refractiveindex.info.
