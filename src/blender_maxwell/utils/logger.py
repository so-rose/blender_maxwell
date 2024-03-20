import logging

LOGGER = logging.getLogger('blender_maxwell')


def get():
	if LOGGER is None:
		# Set Sensible Defaults
		LOGGER.setLevel(logging.DEBUG)
		#FORMATTER = logging.Formatter(
		#	'%(asctime)-15s %(levelname)8s %(name)s %(message)s'
		#)

		# Add Stream Handler
		STREAM_HANDLER = logging.StreamHandler()
		#STREAM_HANDLER.setFormatter(FORMATTER)
		LOGGER.addHandler(STREAM_HANDLER)

	return LOGGER

def set_level(level):
	LOGGER.setLevel(level)
def enable_logfile():
	raise NotImplementedError
def disable_logfile():
	raise NotImplementedError
