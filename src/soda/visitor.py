import collections

from haoda import ir
from soda import core

def get_load_tuple(obj):
  """Get all load references as a tuple.

  Args:
    obj: A haoda.ir.Node object or a soda.core.Tensor object.

  Returns:
    A tuple of all the load references.

  Raises:
    TypeError: If obj is not an IR node or a Tensor.
  """
  def visitor(obj, loads):
    if isinstance(obj, ir.Ref):
      loads.append(obj)
    return obj
  loads = []
  if isinstance(obj, ir.Node):
    obj.visit(visitor, loads)
  elif isinstance(obj, core.Tensor):
    obj.visit_loads(visitor, loads)
  else:
    raise TypeError('argument is not an IR node or a Tensor')
  return tuple(loads)

def get_load_set(obj):
  """Get all unique load references as a tuple.

  Args:
    obj: A haoda.ir.Node object.

  Returns:
    A tuple of all unique loads.

  Raises:
    TypeError: If obj is not an IR node.
  """
  def visitor(obj, loads):
    if isinstance(obj, ir.Ref):
      loads[obj] = None
    return obj
  loads = collections.OrderedDict()
  if isinstance(obj, ir.Node):
    obj.visit(visitor, loads)
  else:
    raise TypeError('argument is not an IR node or a Tensor')
  return tuple(loads)

def get_load_dict(obj):
  """Get all load references as a dict mapping names to lists of loads.

  Args:
    obj: A soda.core.Tensor object.

  Returns:
    A dict mapping accessed tensor names to the corresponding lists of loads.

  Raises:
    TypeError: If obj is not a Tensor.
  """
  def visitor(obj, loads):
    if isinstance(obj, ir.Ref):
      loads.setdefault(obj.name, []).append(obj)
    return obj
  loads = collections.OrderedDict()
  if isinstance(obj, core.Tensor):
    obj.visit_loads(visitor, loads)
  else:
    raise TypeError('argument is not a Tensor')
  return loads

def get_normalize_index(obj) -> tuple:
  """Get the normalize index that will make the least access index 0.

  Args:
    obj: A node or an iterable of nodes.
  Returns:
    Normalize index as a tuple.
  Raises:
    TypeError: If argument is not an ir.Node or an iterable of ir.Nodes.
  """
  if not isinstance(obj, (collections.Iterable, ir.Node)):
    raise TypeError('argument is not an ir.Node or an iterable of ir.Nodes')
  if isinstance(obj, ir.Node):
    obj = (obj,)
  try:
    return min(sum(map(get_load_tuple, obj), ()),
               key=lambda load: tuple(reversed(load.idx))).idx
  except ValueError as e:
    if str(e) == 'min() arg is an empty sequence':
      return ()
    raise e
