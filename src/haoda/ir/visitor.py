import collections

from haoda import ir

def get_dram_refs(obj):
  """Get all DRAM references as a tuple.

  Args:
    obj: A haoda.ir.Node object or an Iterable of haoda.ir.Node objects.

  Returns:
    A tuple of all DRAM references.

  Raises:
    TypeError: If obj is not an IR node or a sequence.
  """
  def visitor(obj, args):
    if isinstance(obj, ir.DRAMRef):
      args.append(obj)
    return obj
  if isinstance(obj, collections.Iterable):
    return sum(map(get_dram_refs, obj), ())
  dram_refs = []
  if isinstance(obj, ir.Node):
    obj.visit(visitor, dram_refs)
  else:
    raise TypeError('argument is not an IR node or a sequence')
  return tuple(dram_refs)

def get_read_fifo_set(module):
  """Get all read FIFOs as a tuple. Each FIFO only appears once.

  Args:
    module: A haoda.ir.Module object.

  Returns:
    A tuple of all FIFOs that are read in the module.

  Raises:
    TypeError: If argument is not a module.
  """
  def visitor(obj, args):
    if isinstance(obj, ir.FIFO):
      args[obj] = None
    return obj
  fifo_loads = collections.OrderedDict()
  if isinstance(module, ir.Module):
    module.visit_loads(visitor, fifo_loads)
  else:
    raise TypeError('argument is not a module')
  return tuple(fifo_loads)
