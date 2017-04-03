import collections
import logging

from haoda.ir.arithmetic import base

_logger = logging.getLogger().getChild(__name__)

def simplify(expr):
  """Simplifies expressions.

  Args:
    expr: A haoda.ir.Node or a sequence of haoda.ir.Node.

  Returns:
    Simplified haoda.ir.Node or sequence.
  """

  if expr is None:
    _logger.debug('None expr, no simplification.')
    return expr

  passes = base.compose(
      base.flatten,
      base.print_tree)

  if isinstance(expr, collections.Iterable):
    return type(expr)(map(passes, expr))

  return passes(expr)
