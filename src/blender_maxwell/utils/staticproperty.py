class staticproperty(property):  # noqa: N801
	"""A read-only variant of `@property` that is entirely static, for use in specific situations.

	The decorated method must take no arguments whatsoever, including `self`/`cls`.

	Examples:
		Use as usual:
		```python
		class Spam:
			@staticproperty
			def eggs():
				return 10

		assert Spam.eggs == 10
		```
	"""

	def __get__(self, *_):
		"""Overridden getter that ignores instance and owner, and just returns the value of the evaluated (static) method.

		Returns:
			The evaluated value of the static method that was decorated.
		"""
		return self.fget()
