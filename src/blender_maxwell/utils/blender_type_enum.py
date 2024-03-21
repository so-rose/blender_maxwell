import enum


class BlenderTypeEnum(str, enum.Enum):
	def _generate_next_value_(name, *_):
		return name


def append_cls_name_to_values(cls):
	# Construct Set w/Modified Member Names
	new_members = {
		name: f'{name}{cls.__name__}'
		for name, member in cls.__members__.items()
	}

	# Dynamically Declare New Enum Class w/Modified Members
	new_cls = enum.Enum(cls.__name__, new_members, type=BlenderTypeEnum)
	new_cls.__module__ = cls.__module__

	# Return New (Replacing) Enum Class
	return new_cls


def wrap_values_in_MT(cls):
	# Construct Set w/Modified Member Names
	new_members = {
		name: f'BLENDER_MAXWELL_MT_{name}'
		for name, member in cls.__members__.items()
	}

	# Dynamically Declare New Enum Class w/Modified Members
	new_cls = enum.Enum(cls.__name__, new_members, type=BlenderTypeEnum)
	new_cls.__module__ = cls.__module__
	new_cls.get_tree = cls.get_tree  ## TODO: This is wildly specific...

	# Return New (Replacing) Enum Class
	return new_cls
