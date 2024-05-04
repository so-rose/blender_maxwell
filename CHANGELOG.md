## v0.1.0 (2024-05-04)

### Feat

- E2E simulation design and analysis.
- Feature-parity with past.
- Finished Gaussian Pulse node.
- Finished Library Medium node.
- Added the Bloch boundary condition.
- Added adiabatic absorber.
- Added BoundConds Node & Fancy PML Node
- Use `canvas.draw()` for plotting.
- Implemented operate math node.
- Complete matplotlib plotting system.
- Safe, practical BLField.
- High-performance math system and depsflow.
- Math nodes (non-working)
- Scientific constant node.
- Robust DataFlowKind w/lazy structures.
- Implemented fit of experim. medium data.
- Added Tidy3D file import node
- Better link/append strategy for GN lookup
- Fixes for cloud tasks, lint run
- Proper visualization pathways
- ManagedObj Semantics
- Re-Implemented Unit System Node (+ other fixes)
- Working logging, before- and after-deps.
- Completely revamped dependency system.
- Demo-grade simulation feedback loop.
- Continue to add features.
- Various features (some very prototype).
- We did it, GeoNodes node w/live update!
- More sockets, nodes, fixes.
- Added accel socket, fixed default units.
- Custom units, def. all SocketType units.
- Registered all nodes.
- Somewhat working addon.

### Fix

- Run `active_kind` updator after socket init.
- Unit conversion of LazyValueRange.
- Inching closer.
- Major streamlining of plot workflow.
- Extract fixes incl. draw, array-copy on export.
- Invalidate cache of removed input sockets.
- Implement explicit no-flow w/FlowSignal
- Crashes on enum changes
- Crashiness of EnumProperty
- BLFields in FilterMath, bug fixes.
- Some renamed FlowKinds (not all)
- Caching now (seems to) work robustly.
- Revalidated cache logic w/KeyedCache.
- A bug and a crash.
- Various critical fixes, field preview
- Case-insensitive constants search
- The rabid __mp_main__ segfault.
- Broken GN unit evaluation
- @base event callbacks now use @events
- Bugs related to geonodes, end-of-chain unit conversion

### Refactor

- applied tooling for predictable lint/fmt/commits
- Factored out flow_kinds.py for clarity.
- Huge simplifications from ExprSocket
- Big breakthrough on Expr socket (non working)
- Big changes to data flow and deps loading
- More changes to docs/layout
- Moved contracts + fixes
- Use cleaner relative import for top-level `utils`
- Fixes and movement.
- Renamed DataFlowKind to FlowKind
- Ran lint fix
- Common SocketDef owner in `sockets.base`
- Revamped serialization (non-working)
- Non-working first-move of serialization logic
- Streamlined graph-update semantics.
- Continuing large-scale alterations.
- Massive architectural changes.
- Far more well-functioning baseline.
- Big categories, structure change.
