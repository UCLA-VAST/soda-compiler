import collections
import copy
import logging
import math

import cached_property

from haoda import util
from haoda.ir import visitor

_logger = logging.getLogger().getChild(__name__)

GRAMMAR = r'''
Bin: /0[Bb][01]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Dec: /\d+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Oct: /0[0-7]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Hex: /0[Xx][0-9a-fA-F]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Int: ('+'|'-')?(Hex|Bin|Oct|Dec);
Float: /(((\d*\.\d+|\d+\.)([+-]?[Ee]\d+)?)|(\d+[+-]?[Ee]\d+))[FfLl]?/;
Num: Float|Int;

Type: FixedType | FloatType;
FixedType: /u?int[1-9]\d*(_[1-9]\d*)?/;
FloatType: /float[1-9]\d*(_[1-9]\d*)?/ | 'float' | 'double' | 'half';

Let: (haoda_type=Type)? name=ID '=' expr=Expr;
Ref: name=ID '(' idx=INT (',' idx=INT)* ')' ('~' lat=Int)?;

Expr: operand=LogicAnd (operator=LogicOrOp operand=LogicAnd)*;
LogicOrOp: '||';

LogicAnd: operand=BinaryOr (operator=LogicAndOp operand=BinaryOr)*;
LogicAndOp: '&&';

BinaryOr: operand=Xor (operator=BinaryOrOp operand=Xor)*;
BinaryOrOp: '|';

Xor: operand=BinaryAnd (operator=XorOp operand=BinaryAnd)*;
XorOp: '^';

BinaryAnd: operand=EqCmp (operator=BinaryAndOp operand=EqCmp)*;
BinaryAndOp: '&';

EqCmp: operand=LtCmp (operator=EqCmpOp operand=LtCmp)*;
EqCmpOp: '=='|'!=';

LtCmp: operand=AddSub (operator=LtCmpOp operand=AddSub)*;
LtCmpOp: '<='|'>='|'<'|'>';

AddSub: operand=MulDiv (operator=AddSubOp operand=MulDiv)*;
AddSubOp: '+'|'-';

MulDiv: operand=Unary (operator=MulDivOp operand=Unary)*;
MulDivOp: '*'|'/'|'%';

Unary: (operator=UnaryOp)* operand=Operand;
UnaryOp: '+'|'-'|'~'|'!';

Operand: cast=Cast | call=Call | ref=Ref | num=Num | var=Var | '(' expr=Expr ')';
Cast: haoda_type=Type '(' expr=Expr ')';
Call: name=FuncName '(' arg=Expr (',' arg=Expr)* ')';
Var: name=ID ('[' idx=Int ']')*;

'''

class Node():
  """A immutable, hashable IR node.
  """
  SCALAR_ATTRS = ()
  LINEAR_ATTRS = ()

  @property
  def ATTRS(self):
    return self.SCALAR_ATTRS + self.LINEAR_ATTRS

  def __init__(self, **kwargs):
    for attr in self.SCALAR_ATTRS:
      setattr(self, attr, kwargs.pop(attr))
    for attr in self.LINEAR_ATTRS:
      setattr(self, attr, tuple(kwargs.pop(attr)))

  def __hash__(self):
    return hash((tuple(getattr(self, _) for _ in self.SCALAR_ATTRS),
                 tuple(tuple(getattr(self, _)) for _ in self.LINEAR_ATTRS)))

  def __eq__(self, other):
    return all(hasattr(other, attr) and
               getattr(self, attr) == getattr(other, attr)
               for attr in self.ATTRS)

  @property
  def c_type(self):
    return util.get_c_type(self.haoda_type)

  @property
  def width_in_bits(self):
    return util.get_width_in_bits(self.haoda_type)

  def visit(self, callback, args=None, pre_recursion=None, post_recursion=None):
    """A general-purpose, flexible, and powerful visitor.

    The args parameter will be passed to the callback callable so that it may
    read or write any information from or to the caller.

    A copy of self will be made and passed to the callback to avoid destructive
    access.

    If a new object is returned by the callback, it will be returned directly
    without recursion.

    If the same object is returned by the callback, if any attribute is
    changed, it will not be recursively visited. If an attribute is unchanged,
    it will be recursively visited.
    """

    def callback_wrapper(callback, obj, args):
      if callback is None:
        return obj
      result = callback(obj, args)
      if result is not None:
        return result
      return obj

    self_copy = copy.copy(self)
    obj = callback_wrapper(callback, self_copy, args)
    if obj is not self_copy:
      return obj
    self_copy = callback_wrapper(pre_recursion, copy.copy(self), args)
    scalar_attrs = {attr: getattr(self_copy, attr).visit(
        callback, args, pre_recursion, post_recursion)
                    if isinstance(getattr(self_copy, attr), Node)
                    else getattr(self_copy, attr)
                    for attr in self_copy.SCALAR_ATTRS}
    linear_attrs = {attr: tuple(_.visit(
        callback, args, pre_recursion, post_recursion)
                                if isinstance(_, Node) else _
                                for _ in getattr(self_copy, attr))
                    for attr in self_copy.LINEAR_ATTRS}

    for attr in self.SCALAR_ATTRS:
      # old attribute may not exist in mutated object
      if not hasattr(obj, attr):
        continue
      if getattr(obj, attr) is getattr(self, attr):
        if isinstance(getattr(obj, attr), Node):
          setattr(obj, attr, scalar_attrs[attr])
    for attr in self.LINEAR_ATTRS:
      # old attribute may not exist in mutated object
      if not hasattr(obj, attr):
        continue
      setattr(obj, attr, tuple(
          c if a is b and isinstance(a, Node) else a
          for a, b, c in zip(getattr(obj, attr), getattr(self, attr),
                             linear_attrs[attr])))
    return callback_wrapper(post_recursion, obj, args)

class Let(Node):
  SCALAR_ATTRS = 'haoda_type', 'name', 'expr'

  def __str__(self):
    result = '{} = {}'.format(self.name, unparenthesize(self.expr))
    if self.haoda_type is not None:
      result = '{} {}'.format(self.haoda_type, result)
    return result

  @property
  def haoda_type(self):
    if self._haoda_type is None:
      return self.expr.haoda_type
    return self._haoda_type

  @haoda_type.setter
  def haoda_type(self, val):
    self._haoda_type = val

  @property
  def c_expr(self):
    return 'const {} {} = {};'.format(self.c_type, self.name,
                                      unparenthesize(self.expr.c_expr))

class Ref(Node):
  SCALAR_ATTRS = 'name', 'lat'
  LINEAR_ATTRS = ('idx',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.idx = tuple(self.idx)
    if not hasattr(self, 'haoda_type'):
      self.haoda_type = None
    # self.lat will be defined in super().__init__(**kwargs)
    # pylint: disable=access-member-before-definition
    if isinstance(self.lat, str):
      self.lat = str2int(self.lat)

  def __str__(self):
    result = '{}({})'.format(self.name, ', '.join(map(str, self.idx)))
    if self.lat is not None:
      result += ' ~{}'.format(self.lat)
    return result

class BinaryOp(Node):
  LINEAR_ATTRS = 'operand', 'operator'
  def __str__(self):
    result = str(self.operand[0])
    for operator, operand in zip(self.operator, self.operand[1:]):
      result += ' {} {}'.format(operator, operand)
    if self.singleton:
      return result
    return parenthesize(result)

  @property
  def haoda_type(self):
  # TODO: derive from all operands
    return self.operand[0].haoda_type

  @property
  def c_expr(self):
    result = self.operand[0].c_expr
    for operator, operand in zip(self.operator, self.operand[1:]):
      result += ' {} {}'.format(operator, operand.c_expr)
    if self.singleton:
      return result
    return parenthesize(result)

  @property
  def singleton(self) -> bool:
    return len(self.operand) == 1

class Expr(BinaryOp):
  pass

class LogicAnd(BinaryOp):
  pass

class BinaryOr(BinaryOp):
  pass

class Xor(BinaryOp):
  pass

class BinaryAnd(BinaryOp):
  pass

class EqCmp(BinaryOp):
  pass

class LtCmp(BinaryOp):
  pass

class AddSub(BinaryOp):
  pass

class MulDiv(BinaryOp):
  pass

class Unary(Node):
  SCALAR_ATTRS = ('operand',)
  LINEAR_ATTRS = ('operator',)
  def __str__(self):
    return ''.join(self.operator)+str(self.operand)

  @property
  def haoda_type(self):
    return self.operand.haoda_type

  @property
  def c_expr(self):
    return ''.join(self.operator)+self.operand.c_expr

class Operand(Node):
  SCALAR_ATTRS = 'cast', 'call', 'ref', 'num', 'var', 'expr'
  def __str__(self):
    for attr in ('cast', 'call', 'ref', 'num', 'var'):
      if getattr(self, attr) is not None:
        return str(getattr(self, attr))
    # pylint: disable=useless-else-on-loop
    else:
      return parenthesize(self.expr)

  @property
  def c_expr(self):
    for attr in ('cast', 'call', 'ref', 'num', 'var'):
      attr = getattr(self, attr)
      if attr is not None:
        if hasattr(attr, 'c_expr'):
          return attr.c_expr
        return str(attr)
    # pylint: disable=useless-else-on-loop
    else:
      return parenthesize(self.expr.c_expr)

  @property
  def haoda_type(self):
    for attr in self.ATTRS:
      val = getattr(self, attr)
      if val is not None:
        if hasattr(val, 'haoda_type'):
          return val.haoda_type
        if attr == 'num':
          if 'u' in val.lower():
            if 'll' in val.lower():
              return 'uint64'
            return 'uint32'
          if 'll' in val.lower():
            return 'int64'
          if 'fl' in val.lower():
            return 'double'
          if 'f' in val.lower() or 'e' in val.lower():
            return 'float'
          if '.' in val:
            return 'double'
          return 'int32'
        return None
    raise util.InternalError('undefined Operand')

class Cast(Node):
  SCALAR_ATTRS = 'haoda_type', 'expr'
  def __str__(self):
    return '{}{}'.format(self.haoda_type, parenthesize(self.expr))

  @property
  def c_expr(self):
    return 'static_cast<{} >{}'.format(self.c_type,
                                       parenthesize(self.expr.c_expr))

class Call(Node):
  SCALAR_ATTRS = ('name',)
  LINEAR_ATTRS = ('arg',)
  def __str__(self):
    return '{}({})'.format(self.name, ', '.join(map(str, self.arg)))

  @property
  def haoda_type(self):
    if self.name in ('select',):
      return self.arg[1].haoda_type
    return self.arg[0].haoda_type

  @property
  def c_expr(self):
    return '{}({})'.format(self.name, ', '.join(_.c_expr for _ in self.arg))

class Var(Node):
  SCALAR_ATTRS = ('name',)
  LINEAR_ATTRS = ('idx',)
  def __str__(self):
    return self.name+''.join(map('[{}]'.format, self.idx))

  @property
  def c_expr(self):
    return self.name+''.join(map('[{}]'.format, self.idx))

class FIFO(Node):
  """A reference to another node in a haoda.ir.Expr.

  This is used to represent a read/write from/to a Module in an output's Expr.
  It replaces Ref in haoda.ir, which is used to represent an element
  reference to a tensor.

  Attributes:
    read_module: Module reading from this FIFO.
    read_lat: int, at what cycle of a pipelined loop it is being read.
    write_module: Module writing to this FIFO.
    write_lat: int, at what cycle of a pipelined loop it is being written.
    depth: int, FIFO depth.
  """
  IMMUTABLE_ATTRS = 'read_module', 'write_module'
  SCALAR_ATTRS = 'read_module', 'read_lat', 'write_module', 'write_lat', 'depth'

  def __init__(self, write_module, read_module,
               depth=None, write_lat=None, read_lat=None):
    super().__init__(write_module=write_module, read_module=read_module,
                     depth=depth, write_lat=write_lat, read_lat=read_lat)

  def __repr__(self):
    return 'fifo[%d]: %s%s => %s%s' % (self.depth, repr(self.write_module),
      '' if self.write_lat is None else ' ~%s'%self.write_lat,
      repr(self.read_module),
      '' if self.read_lat is None else ' ~%s'%self.read_lat)

  def __hash__(self):
    return hash(tuple(getattr(self, _) for _ in self.IMMUTABLE_ATTRS))

  def __eq__(self, other):
    return all(getattr(self, _) == getattr(other, _)
               for _ in type(self).IMMUTABLE_ATTRS)
  @property
  def edge(self):
    return self.write_module, self.read_module

  @property
  def haoda_type(self):
    return self.write_module.exprs[self].haoda_type

  @property
  def c_expr(self):
    return 'from_{}_to_{}'.format(self.write_module.name, self.read_module.name)

class Module():
  """A node in the dataflow graph.

  This is the base class for a dataflow module. It defines the parent (input)
  nodes, children (output) nodes, output expressions, input schedules, and
  output schedules. It also has a name to help identify itself.

  Attributes:
    parents: Set of parent (input) Module.
    children: Set of child (output) Module.
    lets: List of haoda.ir.Let expressions.
    exprs: Dict of {FIFO: haoda.ir.Expr}, stores an output's expression.
  """
  def __init__(self):
    """Initializes attributes into empty list or dict.
    """
    self.parents = []
    self.children = []
    self.lets = []
    self.exprs = collections.OrderedDict()

  @property
  def name(self):
    return 'module_%u' % hash(self)

  @property
  def fifos(self):
    return tuple(self.exprs.keys())

  @property
  def fifo_dict(self):
    return {(self, fifo.read_module): fifo for fifo in self.exprs}

  def fifo(self, dst_node):
    return self.fifo_dict[(self, dst_node)]

  def get_latency(self, dst_node):
    return self.fifo(dst_node).write_lat or 0

  def visit_loads(self, callback, args=None):
    obj = copy.copy(self)
    obj.lets = tuple(_.visit(callback, args) for _ in self.lets)
    obj.exprs = collections.OrderedDict()
    for fifo in self.exprs:
      obj.exprs[fifo] = self.exprs[fifo].visit(callback, args)
    return obj

  @property
  def dram_reads(self):
    return self._interfaces['dram_reads']

  @property
  def dram_writes(self):
    return self._interfaces['dram_writes']

  @property
  def input_fifos(self):
    return self._interfaces['input_fifos']

  @property
  def output_fifos(self):
    return self._interfaces['output_fifos']

  @cached_property.cached_property
  def _interfaces(self):
    # find dram reads
    reads_in_lets = tuple(_.expr for _ in self.lets)
    reads_in_exprs = tuple(self.exprs.values())
    dram_reads = collections.OrderedDict()
    for dram_ref in visitor.get_dram_refs(reads_in_lets + reads_in_exprs):
      for bank in dram_ref.dram:
        dram_reads[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_reads = tuple(dram_reads.values())

    # find dram writes
    writes_in_lets = tuple(_.name for _ in self.lets
                           if not isinstance(_.name, str))
    dram_writes = collections.OrderedDict()
    for dram_ref in visitor.get_dram_refs(writes_in_lets):
      for bank in dram_ref.dram:
        dram_writes[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_writes = tuple(dram_writes.values())

    output_fifos = tuple(_.c_expr for _ in self.exprs)
    input_fifos = tuple(_.c_expr for _ in visitor.get_read_fifo_set(self))


    return {
        'dram_writes' : dram_writes,
        'output_fifos' : output_fifos,
        'input_fifos' : input_fifos,
        'dram_reads' : dram_reads
    }

  def __str__(self):
    return '%s @ 0x%x: %s' % (type(self).__name__, id(self),
      self.__dict__)

  def __repr__(self):
    return '%s @ 0x%x' % (type(self).__name__, id(self))

  def add_child(self, child):
    """Add a child (low level).

    This method only handles children and parents field; lets and exprs are
    not updated.

    Arguments:
      child: Module, child being added
    """
    if child not in self.children:
      self.children.append(child)
    if self not in child.parents:
      child.parents.append(self)

  def bfs_node_gen(self):
    """BFS over descendant nodes.

    This method is a BFS traversal generator over all descendant nodes.
    """
    node_queue = collections.deque([self])
    seen_nodes = {self}
    while node_queue:
      node = node_queue.popleft()
      yield node
      for child in node.children:
        if child not in seen_nodes:
          node_queue.append(child)
          seen_nodes.add(child)

  def dfs_node_gen(self):
    """DFS over descendant nodes.

    This method is a DFS traversal generator over all descendant nodes.
    """
    node_stack = [self]
    seen_nodes = {self}
    while node_stack:
      node = node_stack.pop()
      yield node
      for child in node.children:
        if child not in seen_nodes:
          node_stack.append(child)
          seen_nodes.add(child)

  def tpo_node_gen(self):
    """Traverse descendant nodes in topological order.

    This method is a generator that traverses all descendant nodes in
    topological order.
    """
    nodes = collections.OrderedDict()
    for node in self.bfs_node_gen():
      nodes[node] = len(node.parents)
    while nodes:
      for node in nodes:
        if nodes[node] == 0:
          yield node
          for child in node.children:
            nodes[child] -= 1
          del nodes[node]
          break
      else:
        return

  def bfs_edge_gen(self):
    """BFS over descendant edges.

    This method is a BFS traversal generator over all descendant edges.
    """
    node_queue = collections.deque([self])
    seen_nodes = {self}
    while node_queue:
      node = node_queue.popleft()
      for child in node.children:
        yield node, child
        if child not in seen_nodes:
          node_queue.append(child)
          seen_nodes.add(child)

  def dfs_edge_gen(self):
    """DFS over descendant edges.

    This method is a DFS traversal generator over all descendant edges.
    """
    node_stack = [self]
    seen_nodes = {self}
    while node_stack:
      node = node_stack.pop()
      for child in node.children:
        yield node, child
        if child not in seen_nodes:
          node_stack.append(child)
          seen_nodes.add(child)

  def get_descendants(self):
    """Get all descendant nodes.

    This method returns all descendant nodes as a set.

    Returns:
      Set of descendant Module.
    """
    return {self}.union(*map(Module.get_descendants, self.children))

  def get_connections(self):
    """Get all descendant edges.

    This method returns all descendant edges as a set.

    Returns:
      Set of descendant (src Module, dst Module) tuple.
    """
    return ({(self, child) for child in self.children}
        .union(*map(Module.get_connections, self.children)))


class DelayedRef(Node):
  """A delayed FIFO reference.

  Attributes:
    delay: int
    ref: FIFO
  """
  SCALAR_ATTRS = ('delay', 'ref')
  @property
  def haoda_type(self):
    return self.ref.haoda_type

  def __str__(self):
    return '%s delayed %d' % (self.ref, self.delay)

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash((self.delay, self.ref))

  def __eq__(self, other):
    return all(getattr(self, attr) == getattr(other, attr)
               for attr in ('delay', 'ref'))

  @property
  def buf_name(self):
    return '{ref.c_expr}_delayed_{delay}_buf'.format(**self.__dict__)

  @property
  def ptr(self):
    return '{ref.c_expr}_delayed_{delay}_ptr'.format(**self.__dict__)

  @property
  def ptr_type(self):
    return 'uint%d' % int(math.log2(self.delay)+1)

  @property
  def c_expr(self):
    return '{ref.c_expr}_delayed_{delay}'.format(**self.__dict__)

  @property
  def c_ptr_type(self):
    return util.get_c_type(self.ptr_type)

  @property
  def c_ptr_decl(self):
    return '{} {} = 0;'.format(self.c_ptr_type, self.ptr)

  @property
  def c_buf_ref(self):
    return '{}[{}]'.format(self.buf_name, self.ptr)

  @property
  def c_buf_decl(self):
    return '{} {}[{}];'.format(self.c_type, self.buf_name, self.delay)

  @property
  def c_buf_load(self):
    return '{} = {};'.format(self.c_expr, self.c_buf_ref)

  @property
  def c_buf_store(self):
    return '{} = {};'.format(self.c_buf_ref, self.ref.ref_name)

  @property
  def c_next_ptr_expr(self):
    return '{ptr} < {depth} ? {c_ptr_type}({ptr}+1) : {c_ptr_type}(0)'.format(
        ptr=self.ptr, c_ptr_type=self.c_ptr_type, depth=self.delay-1)

class FIFORef(Node):
  """A FIFO reference.

  Attributes:
    fifo: FIFO it is linked to
    lat: int, at what cycle of a pipelined loop it is being referenced.
    ref_id: int, reference id in the current scope
  Properties:
    c_type: str
    c_expr: str
    haoda_type: str
    ld_name: str
    st_name: str
    ref_name: str
  """
  SCALAR_ATTRS = ('fifo', 'lat', 'ref_id')
  LD_PREFIX = 'fifo_ld_'
  ST_PREFIX = 'fifo_st_'
  REF_PREFIX = 'fifo_ref_'
  def __str__(self):
    return '<%s fifo_ref_%d%s>' % (self.haoda_type, self.ref_id,
                                   '@%s'%self.lat if self.lat else '')

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash((self.lat, self.ref_id))

  def __eq__(self, other):
    return all(getattr(self, attr) == getattr(other, attr)
               for attr in ('lat', 'ref_id'))

  @property
  def haoda_type(self):
    return self.fifo.haoda_type

  @property
  def ld_name(self):
    return '{LD_PREFIX}{ref_id}'.format(**self.__dict__, **type(self).__dict__)

  @property
  def ref_name(self):
    return '{REF_PREFIX}{ref_id}'.format(**self.__dict__, **type(self).__dict__)

  @property
  def c_expr(self):
    return self.ref_name

class DRAMRef(Node):
  """A DRAM reference.

  Attributes:
    haoda_type: str
    dram: [int], DRAM id it is accessing
    var: str, variable name it is accessing
    offset: int
  """
  SCALAR_ATTRS = 'haoda_type', 'dram', 'var', 'offset'
  def __str__(self):
    return 'dram<bank {} {}@{}>'.format(util.lst2str(self.dram),
                                        self.var, self.offset)

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash((self.dram, self.offset))

  def __eq__(self, other):
    return all(getattr(self, attr) == getattr(other, attr)
               for attr in ('dram', 'offset'))
  @property
  def c_expr(self):
    return str(self)

  def dram_buf_name(self, bank):
    assert bank in self.dram, 'unexpected bank {}'.format(bank)
    return 'dram_{}_bank_{}_buf'.format(self.var, bank)

  def dram_fifo_name(self, bank):
    assert bank in self.dram, 'unexpected bank {}'.format(bank)
    return 'dram_{}_bank_{}_fifo'.format(self.var, bank)

class ModuleTrait(Node):
  """A immutable, hashable trait of a dataflow module.

  Attributes:
    lets: tuple of lets
    exprs: tuple of exprs
    template_types: tuple of template types (TODO)
    template_ints: tuple of template ints (TODO)

  Properties:
    loads: tuple of FIFORefs
  """
  LINEAR_ATTRS = ('lets', 'exprs', 'template_types', 'template_ints')

  def __init__(self, node):
    def mutate(obj, loads):
      if isinstance(obj, FIFO):
        if loads:
          if obj not in loads:
            load_id = next(reversed(loads.values())).ref_id+1
          else:
            return loads[obj]
        else:
          load_id = 0
        fifo_ref = FIFORef(fifo=obj, lat=obj.read_lat, ref_id=load_id)
        loads[obj] = fifo_ref
        return fifo_ref
      return obj
    loads = collections.OrderedDict()
    node = node.visit_loads(mutate, loads)
    self.loads = tuple(loads.values())
    super().__init__(lets=tuple(node.lets), exprs=tuple(node.exprs.values()),
                     template_types=tuple(), template_ints=tuple())
    _logger.debug('Signature: %s', self)

  def __repr__(self):
    return '%s(loads: %s, lets: %s, exprs: %s)' % (
        type(self).__name__,
        util.idx2str(self.loads),
        util.idx2str(self.lets),
        util.idx2str(self.exprs))

  @property
  def dram_reads(self):
    return self._interfaces['dram_reads']

  @property
  def dram_writes(self):
    return self._interfaces['dram_writes']

  @property
  def input_fifos(self):
    return self._interfaces['input_fifos']

  @property
  def output_fifos(self):
    return self._interfaces['output_fifos']

  @cached_property.cached_property
  def _interfaces(self):
    # find dram reads
    reads_in_lets = tuple(_.expr for _ in self.lets)
    reads_in_exprs = tuple(self.exprs)
    dram_reads = collections.OrderedDict()
    for dram_ref in visitor.get_dram_refs(reads_in_lets + reads_in_exprs):
      for bank in dram_ref.dram:
        dram_reads[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_reads = tuple(dram_reads.values())

    # find dram writes
    writes_in_lets = tuple(_.name for _ in self.lets
                           if not isinstance(_.name, str))
    dram_writes = collections.OrderedDict()
    for dram_ref in visitor.get_dram_refs(writes_in_lets):
      for bank in dram_ref.dram:
        dram_writes[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_writes = tuple(dram_writes.values())

    output_fifos = tuple('{}{}'.format(FIFORef.ST_PREFIX, idx)
                         for idx, expr in enumerate(self.exprs))
    input_fifos = tuple(_.ld_name for _ in self.loads)

    return {
        'dram_writes' : dram_writes,
        'output_fifos' : output_fifos,
        'input_fifos' : input_fifos,
        'dram_reads' : dram_reads
    }

def make_var(val):
  """Make literal Var from val."""
  return Var(name=val, idx=())

def str2int(s, none_val=None):
  if s is None:
    return none_val
  while s[-1] in 'UuLl':
    s = s[:-1]
  if s[0:2] == '0x' or s[0:2] == '0X':
    return int(s, 16)
  if s[0:2] == '0b' or s[0:2] == '0B':
    return int(s, 2)
  if s[0] == '0':
    return int(s, 8)
  return int(s)

def parenthesize(expr) -> str:
  return '({})'.format(unparenthesize(expr))

def unparenthesize(expr) -> str:
  expr_str = str(expr)
  while expr_str.startswith('(') and expr_str.endswith(')'):
    expr_str = expr_str[1:-1]
  return expr_str

def get_result_type(operand1, operand2, operator):
  for t in ('double', 'float') + sum((('int%d_t'%w, 'uint%d_t'%w)
                                      for w in (64, 32, 16, 8)), tuple()):
    if t in (operand1, operand2):
      return t
  raise util.SemanticError('cannot parse type: %s %s %s' %
    (operand1, operator, operand2))
