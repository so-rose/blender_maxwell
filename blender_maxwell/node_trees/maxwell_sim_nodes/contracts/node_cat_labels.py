from .node_cats import NodeCategory as NC

NODE_CAT_LABELS = {
	# Inputs/
	NC.MAXWELLSIM_INPUTS: "Inputs",
	NC.MAXWELLSIM_INPUTS_IMPORTERS: "Importers",
	NC.MAXWELLSIM_INPUTS_SCENE: "Scene",
	NC.MAXWELLSIM_INPUTS_PARAMETERS: "Parameters",
	NC.MAXWELLSIM_INPUTS_CONSTANTS: "Constants",
	NC.MAXWELLSIM_INPUTS_LISTS: "Lists",
	
	# Outputs/
	NC.MAXWELLSIM_OUTPUTS: "Outputs",
	NC.MAXWELLSIM_OUTPUTS_VIEWERS: "Viewers",
	NC.MAXWELLSIM_OUTPUTS_EXPORTERS: "Exporters",
	NC.MAXWELLSIM_OUTPUTS_PLOTTERS: "Plotters",
	
	# Sources/
	NC.MAXWELLSIM_SOURCES: "Sources",
	NC.MAXWELLSIM_SOURCES_TEMPORALSHAPES: "Temporal Shapes",
	
	# Mediums/
	NC.MAXWELLSIM_MEDIUMS: "Mediums",
	NC.MAXWELLSIM_MEDIUMS_NONLINEARITIES: "Non-Linearities",
	
	# Structures/
	NC.MAXWELLSIM_STRUCTURES: "Structures",
	NC.MAXWELLSIM_STRUCTURES_PRIMITIVES: "Primitives",
	
	# Bounds/
	NC.MAXWELLSIM_BOUNDS: "Bounds",
	NC.MAXWELLSIM_BOUNDS_BOUNDCONDS: "Bound Conds",
	
	# Monitors/
	NC.MAXWELLSIM_MONITORS: "Monitors",
	NC.MAXWELLSIM_MONITORS_NEARFIELDPROJECTIONS: "Near-Field Projections",
	
	# Simulations/
	NC.MAXWELLSIM_SIMS: "Simulations",
	NC.MAXWELLSIM_SIMGRIDAXES: "Sim Grid Axes",
	
	# Utilities/
	NC.MAXWELLSIM_UTILITIES: "Utilities",
	NC.MAXWELLSIM_UTILITIES_CONVERTERS: "Converters",
	NC.MAXWELLSIM_UTILITIES_OPERATIONS: "Operations",
}
