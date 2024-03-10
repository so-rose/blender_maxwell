import types
import tidy3d.web as td_web

AUTHENTICATED = False

def td_auth(api_key: str):
	# Check for API Key
	if api_key:
		msg = "API Key must be defined to authenticate"
		raise ValueError(msg)
	
	# Perform Authentication
	td_web.configure(api_key)
	try:
		td_web.test()
	except:
		msg = "Tidy3D Cloud Authentication Failed"
		raise ValueError(msg)
	
	AUTHENTICATED = True

def is_td_web_authed(force_check: bool = False) -> bool:
	"""Checks whether `td_web` is authenticated, using the cache.
	The result is heuristically accurate.
	
	If accuracy must be guaranteed, an aliveness-check can be performed by setting `force_check=True`.
	This comes at a performance penalty, as a web request must be made; thus, `force_check` is not appropriate for hot-paths like `draw` functions.
	
	If a check is performed
	"""
	global AUTHENTICATED
	
	# Return Cached Authentication
	if not force_check:
		return AUTHENTICATED
	
	# Re-Check Authentication
	try:
		td_web.test()
		AUTHENTICATED = True  ## Guarantee cache value to True.
		return True
	except:
		AUTHENTICATED = False  ## Guarantee cache value to False.
		return False
	
def g_td_web(api_key: str, force_check: bool = False) -> types.ModuleType:
	"""Returns a `tidy3d.web` module object that is already authenticated using the given API key.
	
	The authentication status is cached using a global module-level variable, `AUTHENTICATED`.
	"""
	global AUTHENTICATED
	
	# Check Cached Authentication
	if not is_td_web_authed(force_check=force_check):
		td_auth(api_key)
		
	return td_web
