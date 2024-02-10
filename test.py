import enum

class BlenderTypeEnum(str, enum.Enum):
	def _generate_next_value_(name, start, count, last_values):
		return name


def blender_type_enum(cls):
	# Construct Set w/Modified Member Names
	new_members = {name: f"{name}{cls.__name__}" for name, member in cls.__members__.items()}
	
	# Dynamically Declare New Enum Class w/Modified Members
	new_cls = enum.Enum(cls.__name__, new_members, type=BlenderTypeEnum)
	new_cls.__module__ = cls.__module__
	
	# Return New (Replacing) Enum Class
	return new_cls

@blender_type_enum
class TreeType(enum.Enum):
	MaxwellSim = enum.auto()

@blender_type_enum
class SocketType(enum.Enum):
	MaxwellSource = enum.auto()
	MaxwellMedium = enum.auto()
	MaxwellStructure = enum.auto()
	MaxwellBound = enum.auto()
	MaxwellFDTDSim = enum.auto()

# Demonstration
print(TreeType.MaxwellSim.value)  # Should print "MaxwellSimTreeType"
print(SocketType.MaxwellSource.value)  # Should print "MaxwellSourceSocketType"

