import functools
import re
import types

def copy_func(f):
     g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
     g = functools.update_wrapper(g, f)
     g.__kwdefaults__ = f.__kwdefaults__
     return g