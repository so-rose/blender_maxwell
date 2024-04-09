import enum


class DataFlowAction(enum.StrEnum):
	# Locking
	EnableLock = 'enable_lock'
	DisableLock = 'disable_lock'

	# Value
	DataChanged = 'value_changed'

	# Previewing
	ShowPreview = 'show_preview'
	ShowPlot = 'show_plot'
