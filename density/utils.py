import os
import sys
import copy

class BaseClass(object):
    """
    Base class to be used throughout this package.
    """
    def __init__(self, *args, **kwargs):
        if len(args):
            if isinstance(args[0], self.__class__):
                self.__dict__.update(args[0].__dict__)
                return
            try:
                kwargs = {**args[0], **kwargs}
            except TypeError:
                args = dict(zip(self._defaults, args))
                kwargs = {**args, **kwargs}
        for name, value in self._defaults.items():
            setattr(self, name, value)
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update input attributes."""
        for name, value in kwargs.items():
            if name in self._defaults:
                setattr(self, name, value)
            else:
                raise ValueError('Unknown argument {}; supports {}'.format(name, list(self._defaults)))

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self):
        return self.__copy__()

    def __setstate__(self, state, load=False):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state, load=False):
        new = cls.__new__(cls)
        new.__setstate__(state, load=load)
        return new

    def save(self, filename):
        self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state, load=True)
        return new