# coding: utf-8

import six
import copy

class ArgScope(object):
  STACK = []
  @staticmethod
  def get_args():
    return copy.deepcopy(ArgScope.STACK[-1]) if len(ArgScope.STACK) > 0 else {}
  @staticmethod
  def _print():
    print('-'*20)
    if len(ArgScope.STACK) == 0:
      print('<empty>')
      return
    for i, args in enumerate(ArgScope.STACK):
      for key, value in six.iteritems(args):
        print('{} {}: {}'.format(i, key, value))
  @staticmethod
  def append(scope, key, value):
    key = key.func_name
    if key in scope:
      scope[key].update(value.copy())
    else:
      scope[key] = value.copy()
  def __init__(self, args):
    self.args = args
  def __enter__(self):
    scope = ArgScope.get_args()
    for keys, value in six.iteritems(self.args):
      if type(keys) == tuple:
        for key in keys:
          ArgScope.append(scope, key, value)
      else:
        ArgScope.append(scope, keys, value)
    ArgScope.STACK.append(scope)
  def __exit__(self, exc_type, exc_value, traceback):
    ArgScope.STACK.pop()

def arg_scope(*args, **kwargs):
  return ArgScope(*args, **kwargs)

def add_args(func):
  name = func.func_name if hasattr(func, 'func_name') else func.__name__
  def _func(*args, **kwargs):
    scope_kwarg = ArgScope.get_args().get(name, {})
    scope_kwarg.update(kwargs)
    return func(*args, **scope_kwarg)
  _func.func_name = name
  return _func