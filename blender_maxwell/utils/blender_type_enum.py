import enum

class BlenderTypeEnum(str, enum.Enum):
	def _generate_next_value_(name, start, count, last_values):
		return name

def append_cls_name_to_values(cls):
	# Construct Set w/Modified Member Names
	new_members = {name: f"{name}{cls.__name__}" for name, member in cls.__members__.items()}
	
	# Dynamically Declare New Enum Class w/Modified Members
	new_cls = enum.Enum(cls.__name__, new_members, type=BlenderTypeEnum)
	new_cls.__module__ = cls.__module__
	
	# Return New (Replacing) Enum Class
	return new_cls
