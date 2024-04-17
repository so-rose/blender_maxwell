# Projects / Plugins
## Larger Architectural Changes
[ ] Dedicated way of generating properties for sockets and nodes, incl. helping make better (less boilerplatey) use of callbacks
	- Perhaps, we should also go 100% custom `PropertyGroup`, to the point that we have `nodes`, `sockets` and `props`.
	- This would allow far simplified sockets (the more complex kinds), and help standardize use of ex. units in node properties.
	- Having a dedicated base class for custom props would help avoid issues like forgetting to run `self.sync_prop` on every goddamn update method in every goddamn socket.
[ ] Dedicated way of handling node-specific operators without all the boilerplate.

## Field Data
[ ] Directly dealing with field data, instead of having field manipulations be baked into viz node(s).
[ ] Yee Cell Data as Attributes on By-Cell Point Cloud w/GeoNodes Integrations
- In effect, when we have xarray data defined based on Yee Cells ex. Poynting vector coordinates, let's import this to Blender as a simple point cloud centered at each cell and grant each an attribute corresponding to the data.
- What we can then do is use vanilla GeoNodes to ex. read the vector attribute, and draw small arrow meshes (maybe resampled which auto-interpolates the field values) from each point, thus effectively visualizing . vector fields and many other fun things.
- Of course, this is no good for volume cell data - but we can just overlay the raw volume cell data as we please. We can also, if we're sneaky, deal with our volume data as points as far as we can, and then finally do a "points to volume" type deal to make it sufficiently "fluffy/cloudy".
- I wonder if we could use the Attribute node in the shader editor to project interpolated values from points, onto a ex. plane mesh, in a way that would also be visualizable in the viewport.

## Tidy3D Features
[ ] Symmetry for Performance
- [ ] Implement <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/Symmetry.html>
[ ] Dispersive Model Fitting
[ ] Scattering Matrix Calculator
[ ] Resonance Finder
[ ] Adjoint Optimization
[ ] Design Space Exploration / Parameterization

## Preview Semantics
[ ] Node tree header toggle that toggles a modal operator on and off, which constantly checks the context of the selected nodes, and tries to `bl_select` them (which in turn, should cause the node base class to `bl_select` shit inside).
- Shouldn't survive a file save; always startup with this thing off.
[ ] Custom gizmos attached to preview toggles!
- There is a WIP for GN-driven gizmos: <https://projects.blender.org/blender/blender/pulls/112677>
- Probably best to wait for that, then just add gizmos to existing driven GN trees, as opposed to unholy OGL spaghetti.
[ ] Node-ManagedObj Selection binding
- BL to Node:
	- Trigger: The post-depsgraph handler seems appropriate.
	- Input: Read the object location (origin), using a unit system.
	- Output: Write the input socket value.
	- Condition: Input socket is unlinked. (If it's linked, then lock the object's position. Use sync_link_added() for that)
- Node to BL:
	- Trigger: "Report" event on an input socket that the managed object declares reliance on.
	- Input: The input socket value (linked or unlinked)
	- Output: The object location (origin), using a unit system.

## Parametric Geometry UX
[ ] Consider allowing a mesh attribute (set in ex. geometry node) to specify the name of a medium.
- This allows assembling complex multi-medium structures in one geonodes tree.
- This should result in the spawning of several Medium input sockets in the GeoNodes structure node, named as the attributes are.
- The GeoNodes structure node should then output as array-like TriMeshes, for which mediums are correctly defined.

## Alternative Engines
[ ] Heat Solver
[ ] MEEP integration (<https://meep.readthedocs.io/en/latest/>)
- The main boost would be if we could setup a MEEP simulation entirely from a td.Simulation object.
