from collections import defaultdict, UserList

class DottableDefaultDict(defaultdict):
    """
    A defaultdict that allows you to access the values with dot notation
    The defaultdict lets you access values that are not yet set, defaulting to the value of the default_factory
    This works not just for dict-access as the normal defaultdict, but also by dot-access
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, __name: str) -> any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return self[__name]

    def __str__(self):
        return str({k: eval(str(v)) for k, v in self.items()})

    @classmethod
    def Nest(cls):
        """ A cursed method that makes an infinitely nested defaultdict """
        return cls(cls.Nest)


class DivDotDefDict(DottableDefaultDict):
    """
    A dottable defaultdict that allows you to divide all values by a number
    """
    def __truediv__(self, other):
        for key in self:
            self[key] /= other
        return self


class DivDefDict(defaultdict):
    """
    A defaultdict that allows you to divide all values by a number
    """
    def __truediv__(self, other):
        for key in self:
            self[key] /= other
        return self


class DefPopList(UserList):
    """
    Subclass of list that allows you to 
        set a default value for pop if 
        the index is out of range
    """
    def pop(self, index, default=None):
        try:
            return super().pop(index)
        except IndexError:
            if default is None:
                raise
            return default
