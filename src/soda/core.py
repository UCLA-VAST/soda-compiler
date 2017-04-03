import collections
import copy
import itertools
import logging
import operator

import cached_property

from haoda import ir
from haoda import util
from haoda.ir import arithmetic
from soda import dataflow
from soda import grammar
from soda import util as soda_util
from soda import visitor
from soda import mutator

_logger = logging.getLogger().getChild(__name__)

class Tensor():
  """A tensor that corresponse to an input, local, or output.

  This class is used in the high-level DAG for stencil dependency analysis.
  Each tensor either is an input tensor, or has at least 1 parent tensor, which
  will be used to generate this tensor. Meanwhile, each tensor either is an
  output tensor, or has at least 1 child tensor, which will be computed using
  this tensor.

  Attributes:
    haoda_type: str, type of the tensor element.
    parents: Dict from str of name of Tensor to Tensor.
    children: Dict from str of name of Tensor to Tensor.
    st_ref: Ref, name, index, and latency stored.
    offset: int, shift offset in terms of data elements
    lets: Lets of computation.
    expr: Expr of computation.
    ld_refs: Dict from str of name to dict of Ref loaded.
    ld_delays: Dict from str of name to extra delay of the input.

  Property:
    name: str, unique in each SODA program.
    st_offset: int, stencil offset in terms of data elements.
    st_idx, Tuple of int, the index referenced by its parent stage.
    ld_indices: Dict from str of name to dict of accessed indices of the input.
    ld_offsets: Dict from str of name to dict of offsets of the input.
  """
  def __init__(self, stmt, tile_size):
    self.haoda_type = stmt.haoda_type
    self._tile_size = tile_size
    if isinstance(stmt, grammar.LocalStmtOrOutputStmt):
      self.st_ref = copy.copy(stmt.ref)
      self.st_ref.parent = self
      self.lets = stmt.let
      self.expr = stmt.expr
    elif isinstance(stmt, grammar.InputStmt):
      self._name = stmt.name
      self.st_ref = None
      self.lets = []
      self.expr = None
    else:
      raise util.InternalError('cannot initialize a Tensor from %s' %
                               type(stmt))
    _logger.debug('tensor initialized from stmt `%s`', stmt)
    # pylint: disable=protected-access
    _logger.debug('                   at tx position %d', stmt._tx_position)

    # these fields are to be set externally
    self.st_delay = 0
    self.parents = collections.OrderedDict()
    self.children = collections.OrderedDict()
    self.ld_refs = collections.OrderedDict()
    self.ld_delays = collections.OrderedDict()

  @property
  def name(self):
    if self.st_ref is not None:
      return self.st_ref.name
    return self._name

  @property
  def st_idx(self):
    if self.st_ref is not None:
      return self.st_ref.idx
    return (0,)*len(self._tile_size)

  @property
  def st_offset(self):
    return soda_util.serialize(self.st_idx, self._tile_size) + self.st_delay

  @cached_property.cached_property
  def ld_indices(self):
    return collections.OrderedDict(
        (name, collections.OrderedDict((ref.idx, ref) for ref in refs))
        for name, refs in self.ld_refs.items())

  @cached_property.cached_property
  def ld_offsets(self):
    return collections.OrderedDict(
      (name, collections.OrderedDict(
        (soda_util.serialize(ref.idx, self._tile_size), ref) for ref in refs))
      for name, refs in self.ld_refs.items())

  @property
  def c_type(self):
    return util.get_c_type(self.haoda_type)

  def propagate_type(self):
    if self.expr is None:
      return

    var_types = {}
    # pylint: disable=access-member-before-definition
    for let in self.lets:
      var_types[let.name] = let.haoda_type

    def visit_haoda_type(obj, args):
      if obj.haoda_type is None:
        if isinstance(obj, ir.Var):
          obj.haoda_type = var_types[obj.name]
      return obj

    self.lets = tuple(_.visit(visit_haoda_type) for _ in self.lets)
    self.expr = self.expr.visit(visit_haoda_type)
    self.st_ref = self.st_ref.visit(visit_haoda_type)

  def mutate(self, callback, args=None):
    self.lets = tuple(_.visit(callback, args) for _ in self.lets)
    self.expr = self.expr.visit(callback, args)
    self.st_ref = self.st_ref.visit(callback, args)

  def visit_loads(self, callback, args=None):
    for let in self.lets:
      let.visit(callback, args)
    self.expr.visit(callback, args)

  def __str__(self):
    return '''Tensor
  {haoda_type}: {name} = {expr}
  store: {st_ref} with delay {st_delay}
  parents: {parents}
  children: {children}'''.format(
      name=self.name, haoda_type=self.haoda_type, expr=self.expr,
      parents=util.idx2str(self.parents), children=util.idx2str(self.children),
      st_ref=str(self.st_ref), st_delay=self.st_delay)

  def is_output(self):
    return len(self.children) == 0

  def is_input(self):
    return len(self.parents) == 0

  def is_producer(self):
    return not self.is_output()

  def is_consumer(self):
    return not self.is_input()

class Stencil():
  """
  Attributes:
    iterate: int, number of iteration to implement.
    burst_width: int, width of bits for DRAM burst access.
    app_name: str, application's name.
    tile_size: List of int.
    unroll_factor: int.
    dim: int.
    param_stmts: List of ParamStmt.
    input_stmts: List of InputStmt.
    local_stmts: List of LocalStmt.
    output_stmts: List of OutputStmt.

  Cached properties:
    tensors: Dict from str of name to Tensor.
    input_names: Tuple of str, names of input tensors.
    param_names: Tuple of str, names of param tensors.
    local_names: Tuple of str, names of local tensors.
    output_names: Tuple of str, names of output tensors.
  """
  def __init__(self, **kwargs):
    self.iterate = kwargs.pop('iterate')
    if self.iterate < 1:
      raise util.SemanticError('cannot iterate %d times' % self.iterate)
    # platform determined
    self.burst_width = kwargs.pop('burst_width')
    # application determined
    self.app_name = kwargs.pop('app_name')
    # parameters that can be explored
    self.tile_size = tuple(kwargs.pop('tile_size'))
    self.unroll_factor = kwargs.pop('unroll_factor')
    # stage-independent
    self.dim = kwargs.pop('dim')
    self.param_stmts = kwargs.pop('param_stmts')
    # stage-specific
    self.input_stmts = kwargs.pop('input_stmts')
    self.local_stmts = kwargs.pop('local_stmts')
    self.output_stmts = kwargs.pop('output_stmts')

    if 'dram_in' in kwargs:
      dram_in = kwargs.pop('dram_in')
      if dram_in is not None:
        if ':' in dram_in:
          input_stmt_map = {_.name : _ for _ in self.input_stmts}
          for dram_map in dram_in.split('^'):
            var_name, bank_list = dram_map.split(':')
            if var_name not in input_stmt_map:
              raise util.SemanticError('no input named `{}`'.format(var_name))
            input_stmt_map[var_name].dram = tuple(map(int,
                                                      bank_list.split('.')))
        else:
          for input_stmt in self.input_stmts:
            input_stmt.dram = tuple(map(int, dram_in.split('.')))

    if 'dram_out' in kwargs:
      dram_out = kwargs.pop('dram_out')
      if dram_out is not None:
        if ':' in dram_out:
          output_stmt_map = {_.name : _ for _ in self.output_stmts}
          for dram_map in dram_out.split(','):
            var_name, bank_list = dram_map.split(':')
            if var_name not in output_stmt_map:
              raise util.SemanticError('no output named `{}`'.format(var_name))
            output_stmt_map[var_name].dram = tuple(map(int,
                                                       bank_list.split('.')))
        else:
          for output_stmt in self.output_stmts:
            output_stmt.dram = tuple(map(int, dram_out.split('.')))

    if self.iterate > 1:
      if len(self.input_stmts) != len(self.output_stmts):
        raise util.SemanticError(
          'number of input tensors must be the same as output if iterate > 1 '
          'times, currently there are %d input(s) but %d output(s)' %
          (len(self.input_stmts), len(self.output_stmts)))
      if self.input_types != self.output_types:
        raise util.SemanticError(
          'input must have the same type(s) as output if iterate > 1 '
          'times, current input has type %s but output has type %s' %
          (util.lst2str(self.input_types), util.lst2str(self.output_types)))
      _logger.debug('pipeline %d iterations of [%s] -> [%s]' % (self.iterate,
        ', '.join('%s: %s' % (stmt.haoda_type, stmt.name)
                  for stmt in self.input_stmts),
        ', '.join('%s: %s' % (stmt.haoda_type, stmt.name)
                  for stmt in self.output_stmts)))

    for stmt in itertools.chain(self.local_stmts, self.output_stmts):
      _logger.debug('simplify %s', stmt.name)
      stmt.expr = arithmetic.simplify(stmt.expr)
      stmt.let = arithmetic.simplify(stmt.let)

    # soda frontend successfully parsed
    # triggers cached property
    # replicate tensors for iterative stencil
    # pylint: disable=pointless-statement
    self.tensors
    _logger.debug('producer tensors: [%s]',
                  ', '.join(tensor.name for tensor in self.producer_tensors))
    _logger.debug('consumer tensors: [%s]',
                  ', '.join(tensor.name for tensor in self.consumer_tensors))

    # TODO: build Ref table and Var table
    # generate reuse buffers and get haoda nodes
    # pylint: disable=pointless-statement
    self.dataflow_super_source
    _logger.debug('dataflow: %s', self.dataflow_super_source)

    _logger.debug('module table: %s', dict(self.module_table))
    _logger.debug('module traits: %s', self.module_traits)

  @cached_property.cached_property
  def dataflow_super_source(self):
    return dataflow.create_dataflow_graph(self)

  @property
  def module_table(self):
    return self.dataflow_super_source.module_table

  @property
  def module_traits(self):
    return self.dataflow_super_source.module_traits

  @cached_property.cached_property
  def input_types(self):
    return tuple(tensor.haoda_type for tensor in self.input_stmts)

  @cached_property.cached_property
  def param_types(self):
    return tuple(tensor.haoda_type for tensor in self.param_stmts)

  @cached_property.cached_property
  def local_types(self):
    return tuple(tensor.haoda_type for tensor in self.local_stmts)

  @cached_property.cached_property
  def output_types(self):
    return tuple(tensor.haoda_type for tensor in self.output_stmts)

  @cached_property.cached_property
  def input_names(self):
    return tuple(stmt.name for stmt in self.input_stmts)

  @cached_property.cached_property
  def param_names(self):
    return tuple(stmt.name for stmt in self.param_stmts)

  @cached_property.cached_property
  def local_names(self):
    return tuple(stmt.name for stmt in self.local_stmts)

  @cached_property.cached_property
  def output_names(self):
    return tuple(stmt.name for stmt in self.output_stmts)

  @cached_property.cached_property
  def symbol_table(self):
    """Constructs a mapping from a tensor's name to its type.

    Returns:
      tensor_types: dict from name (str) to haoda_type (str).
    """
    tensor_types = {}
    for name, haoda_type in zip(self.input_names, self.input_types):
      tensor_types[name] = haoda_type
    for name, haoda_type in zip(self.local_names, self.local_types):
      tensor_types[name] = haoda_type
    for name, haoda_type in zip(self.output_names, self.output_types):
      tensor_types[name] = haoda_type
    return tensor_types

  @cached_property.cached_property
  def tensors(self):
    """Constructs high-level DAG and creates the tensors.

    Returns:
      An collections.OrderedDict mapping a tensor's name to the tensor.
    """
    # TODO: check for name conflicts
    tensor_map = collections.OrderedDict()
    for stmt in self.input_stmts:
      tensor = Tensor(stmt, self.tile_size)
      tensor_map[stmt.name] = tensor

    def name_in_iter(name, iteration):
      if name in self.input_names:
        if iteration > 0:
          return name+'_iter%d' % iteration
        return name
      if name in self.output_names:
        if iteration < self.iterate-1:
          return (self.input_names[self.output_names.index(name)]+
                  '_iter%d' % (iteration+1))
        return name
      if name in self.local_names:
        if iteration > 0:
          return name+'_iter%d' % iteration
        return name
      if name in self.param_names:
        return name
      raise util.InternalError('unknown name: %s' % name)

    for iteration in range(self.iterate):
      _logger.debug('iterate %s', iteration)
      _logger.debug('map: %s', self.symbol_table)
      def mutate_name_callback(obj, mutated):
        if isinstance(obj, ir.Ref):
          obj.haoda_type = self.symbol_table[obj.name]
          # pylint: disable=cell-var-from-loop
          obj.name = name_in_iter(obj.name, iteration)
        return obj
      tensors = []
      for stmt in itertools.chain(self.local_stmts, self.output_stmts):
        tensor = Tensor(stmt.visit(mutate_name_callback), self.tile_size)
        loads = visitor.get_load_tuple(tensor)
        norm_idx = tuple(min(load.idx[d] for load in loads
                             if load.name not in self.param_names)
                         for d in range(self.dim))
        if any(norm_idx):
          _logger.debug('normalize index of %s: (%s)',
                        tensor.name, ', '.join(map(str, norm_idx)))
          mutator.shift(tensor, norm_idx, excluded=self.param_names)
        tensor_map[tensor.name] = tensor
        tensors.append(tensor)

      for tensor in tensors:
        _logger.debug('%s', tensor)

      for tensor in tensors:
        tensor.propagate_type()
        loads = visitor.get_load_dict(tensor)
        for parent_name, ld_refs in loads.items():
          ld_refs = sorted(ld_refs, key=lambda ref: soda_util.serialize(
              ref.idx, self.tile_size))
          parent_tensor = tensor_map[parent_name]
          parent_tensor.children[tensor.name] = tensor
          tensor.parents[parent_name] = parent_tensor
          tensor.ld_refs[parent_name] = ld_refs

    # high-level DAG construction finished
    for tensor in tensor_map.values():
      if tensor.name in self.input_names:
        _logger.debug('<input tensor>: %s', tensor)
      elif tensor.name in self.output_names:
        _logger.debug('<output tensor>: %s', tensor)
      else:
        _logger.debug('<local tensor>: %s', tensor)
    return tensor_map

  @cached_property.cached_property
  def chronological_tensors(self):
    """Computes the offsets of tensors.

    Returns:
      A list of Tensor, in chronological order.
    """
    _logger.info('calculate tensor offsets')
    processing_queue = collections.deque(list(self.input_names))
    processed_tensors = set(self.input_names)
    chronological_tensors = list(map(self.tensors.get, self.input_names))
    for tensor in chronological_tensors:
      _logger.debug('tensor <%s> is at offset %d' %
                    (tensor.name, tensor.st_offset))
    _logger.debug('processing queue: %s', processing_queue)
    _logger.debug('processed_tensors: %s', processed_tensors)
    while processing_queue:
      tensor = self.tensors[processing_queue.popleft()]
      _logger.debug('inspecting tensor %s\'s children' % tensor.name)
      for child in tensor.children.values():
        if ({x.name for x in child.parents.values()} <= processed_tensors
          and child.name not in processed_tensors):
          # good, all inputs are processed
          # can determine offset of current tensor
          _logger.debug(
            'input%s for tensor <%s> (i.e. %s) %s processed',
              '' if len(child.parents) == 1 else 's',
              child.name,
              ', '.join([x.name for x in child.parents.values()]),
              'is' if len(child.parents) == 1 else 'are')
          stage_offset = soda_util.serialize(child.st_idx, self.tile_size)

          # synchronization check
          def sync(tensor, offset):
            if tensor is None:
              return offset
            _logger.debug('index of tensor <%s>: %s',
                          tensor.name, tensor.st_idx)
            stage_offset = soda_util.serialize(tensor.st_idx, self.tile_size)
            _logger.debug('offset of tensor <%s>: %d',
                          tensor.name, stage_offset)
            loads = visitor.get_load_dict(tensor)
            for name in loads:
              loads[name] = tuple(ref.idx for ref in loads[name])
            _logger.debug('loads: %s', ', '.join(
                '%s@%s' % (name, util.lst2str(map(util.idx2str, indices)))
                for name, indices in loads.items()))
            for n in loads:
              loads[n] = soda_util.serialize_iter(loads[n], self.tile_size)
            for l in loads.values():
              l[0], l[-1] = (stage_offset - max(l), stage_offset - min(l))
              del l[1:-1]
              if len(l) == 1:
                l.append(l[-1])
            _logger.debug(
                'load offset range in tensor %s: %s', tensor.name, '{%s}' % (
                    ', '.join('%s: [%d:%d]' % (n, *v)
                              for n, v in loads.items())))
            for parent in tensor.parents.values():
              tensor_distance = next(reversed(tensor.ld_offsets[parent.name]))
              _logger.debug('tensor distance: %s', tensor_distance)
              _logger.debug(
                'want to access tensor <%s> at offset [%d, %d] '
                'to generate tensor <%s> at offset %d',
                  parent.name, offset+loads[parent.name][0],
                  offset+loads[parent.name][-1], tensor.name, offset)
              tensor_offset = (parent.st_delay+tensor_distance-stage_offset)
              if offset < tensor_offset:
                _logger.debug(
                  'but tensor <%s> won\'t be available until offset %d',
                  parent.name, tensor_offset)
                offset = tensor_offset
                _logger.debug('need to access tensor <%s> at offset [%d, %d] '
                              'to generate tensor <%s> at offset %d',
                              parent.name, offset+loads[parent.name][0],
                              offset+loads[parent.name][-1], tensor.name,
                              offset)
            return offset

          _logger.debug('intend to generate tensor <%s> at offset %d',
                        child.name, child.st_delay)
          synced_offset = sync(child, child.st_delay)
          _logger.debug('synced offset: %s', synced_offset)
          child.st_delay = synced_offset
          _logger.debug('decide to generate tensor <%s> at offset %d',
                        child.name, child.st_delay)

          # add delay
          for sibling in child.parents.values():
            delay = child.st_delay - (sibling.st_delay +
                list(child.ld_offsets[sibling.name].keys())[-1] - stage_offset)
            if delay > 0:
              _logger.debug(
                'tensor %s arrives at tensor <%s> at offset %d < %d; '
                'add %d delay', sibling.name, child.name,
                sibling.st_delay+next(reversed(
                    child.ld_offsets[sibling.name]))-stage_offset,
                child.st_delay, delay)
            else:
              _logger.debug(
                'tensor %s arrives at tensor <%s> at offset %d = %d; good',
                sibling.name, child.name, sibling.st_delay+next(reversed(
                  child.ld_offsets[sibling.name]))-stage_offset,
                child.st_delay)
            child.ld_delays[sibling.name] = max(delay, 0)
            _logger.debug('set delay of |%s <- %s| to %d' %
              (child.name, sibling.name, child.ld_delays[sibling.name]))

          processing_queue.append(child.name)
          processed_tensors.add(child.name)
          chronological_tensors.append(child)
        else:
          for parent in tensor.parents.values():
            if parent.name not in processed_tensors:
              _logger.debug('tensor %s requires tensor <%s> as an input',
                            tensor.name, parent.name)
              _logger.debug('but tensor <%s> isn\'t processed yet',
                            parent.name)
              _logger.debug('add %s to scheduling queue',
                            parent.name)
              processing_queue.append(parent.name)

    _logger.debug('tensors in insertion order: [%s]',
                  ', '.join(map(str, self.tensors)))
    _logger.debug('tensors in chronological order: [%s]',
                  ', '.join(t.name for t in chronological_tensors))

    for tensor in self.tensors.values():
      for name, indices in tensor.ld_indices.items():
        _logger.debug('stage index: %s@%s <- %s@%s',
                      tensor.name, util.idx2str(tensor.st_idx),
                      name, util.lst2str(util.idx2str(idx) for idx in indices))
    for tensor in self.tensors.values():
      if tensor.is_input():
        continue
      _logger.debug('stage expr: %s = %s', tensor.st_ref, tensor.expr)
    for tensor in self.tensors.values():
      for name, offsets in tensor.ld_offsets.items():
        _logger.debug('stage offset: %s@%d <- %s@%s',
                      tensor.name, soda_util.serialize(tensor.st_idx,
                                                  self.tile_size),
                      name, util.lst2str(offsets))
    for tensor in self.tensors.values():
      for name, delay in tensor.ld_delays.items():
        _logger.debug('stage delay: %s <- %s delayed %d' %
                (tensor.name, name, delay))

    return chronological_tensors

  @cached_property.cached_property
  def input_partition(self):
    pixel_width_i = sum(self.pixel_width_i)
    if self.burst_width/pixel_width_i*self.dram_bank/2 > self.unroll_factor/2:
      return int(self.burst_width/pixel_width_i*self.dram_bank/2)
    return int(self.unroll_factor/2)

  @cached_property.cached_property
  def output_partition(self):
    pixel_width_o = sum(self.pixel_width_o)
    if self.burst_width/pixel_width_o*self.dram_bank/2 > self.unroll_factor/2:
      return int(self.burst_width/pixel_width_o*self.dram_bank/2)
    return int(self.unroll_factor/2)

  @cached_property.cached_property
  def pixel_width_i(self):
    return list(map(util.get_width_in_bits, self.input_stmts))

  @cached_property.cached_property
  def pixel_width_o(self):
    return list(map(util.get_width_in_bits, self.output_stmts))

  @cached_property.cached_property
  def producer_tensors(self):
    return tuple(filter(Tensor.is_producer, self.tensors.values()))

  @cached_property.cached_property
  def consumer_tensors(self):
    return tuple(filter(Tensor.is_consumer, self.tensors.values()))

  # return [Tensor, ...]
  def _get_parent_tensors_for(self, node):
    return {x: self.tensors[x]
        for x in {x.name for x in node.get_loads()
              if x.name not in self.extra_params}}

  # return {name: [(idx, ...), ...]}
  def _get_window_for(self, node):
    loads = node.get_loads() # [Load, ...]
    load_names = {l.name for l in loads
            if l.name not in self.extra_params}
    windows = {name: sorted({l.idx for l in loads if l.name == name},
                key=lambda x: soda_util.serialize(x, self.tile_size))
           for name in load_names}
    _logger.debug('window for %s@(%s) is %s' %
      (node.name, ', '.join(map(str, node.expr[0].idx)), windows))
    return windows

  # return [StageExpr, ...]
  def _get_expr_for(self, node):
    if isinstance(node, grammar.Output):
      return node.expr
    if isinstance(node, grammar.Local):
      return node.expr
    raise util.SemanticError('cannot get expression for %s' % str(type(node)))

  @cached_property.cached_property
  def reuse_buffers(self):
    """Constructs the reuse buffers.

    Returns:
      A dict mapping a tensor's name to its reuse buffers.
    """
    unroll_factor = self.unroll_factor
    self._reuse_buffer_lengths = {}
    reuse_buffers = {}
    for tensor in self.producer_tensors:
      reuse_buffer = _get_reuse_buffer(self.tile_size, tensor, unroll_factor)
      reuse_buffer_length = {}
      reuse_buffers[tensor.name] = reuse_buffer
      self._reuse_buffer_lengths[tensor.name] = reuse_buffer_length
      first = [True]*unroll_factor
      for start, end in reuse_buffer[1:]:
        if first[start%unroll_factor]:
          first[start%unroll_factor] = False
          if start >= unroll_factor:
            reuse_buffer_length[end] = end//unroll_factor
            continue
        reuse_buffer_length[end] = (end-start)//unroll_factor
    return reuse_buffers

  @cached_property.cached_property
  def all_points(self):
    all_points = {}
    for tensor in self.producer_tensors:
      all_points[tensor.name] = _get_points(self.tile_size, tensor,
                                            self.unroll_factor)
    return all_points

  @cached_property.cached_property
  def next_fifo(self):
    """Constructs the next fifo offset mapping.

    Returns:
      A dict mapping a tensor's name and offset to the next offset.
    """
    next_fifo = {}
    for name, reuse_buffer in self.reuse_buffers.items():
      next_fifo[name] = {}
      for start, end in reuse_buffer[1:]:
        if start < end:
          next_fifo[name][start] = end
    _logger.debug('next_fifo: %s' % next_fifo)
    return next_fifo

  @cached_property.cached_property
  def reuse_buffer_lengths(self):
    """Constructs the reuse buffer lengths.

    Returns:
      A dict mapping a tensor's name to its reuse buffers' lengths.
    """
    # pylint: disable=pointless-statement
    self.reuse_buffers
    return self._reuse_buffer_lengths

def _get_reuse_chains(tile_size, tensor, unroll_factor):
  """Generates reuse chains for a Tensor.

  Generates reuse chains for a Tensor under the given tile size and unroll
  factor.

  Args:
    tile_size: An iterable representing the tile size in each dimension.
    tensor: A Tensor to which the reuse chains belongs.
    unroll_factor: An int representing the unroll factor.

  Returns:
    A list of tuples where each tuple represents a reuse chain and each
    element of the tuple represents the offset from the lastest input.
  """

  _logger.debug('get reuse chains of tensor %s', tensor.name)

  def unroll_offsets(offsets, child):
    unrolled_offsets = set()
    for unroll_idx in range(unroll_factor):
      for offset in offsets:
        unrolled_offsets.add(max(offsets) + unroll_idx - offset +
                             child.ld_delays[tensor.name])
    return unrolled_offsets

  A_dag = set()
  for child in tensor.children.values():
    A_dag |= unroll_offsets(
      soda_util.serialize_iter(child.ld_indices[tensor.name], tile_size), child)
  _logger.debug('A† of tensor %s: %s', tensor.name, A_dag)

  chains = []
  for chain_idx in reversed(range(unroll_factor)):
    chains.append(tuple(sorted(
      offset for offset in A_dag if offset % unroll_factor == chain_idx)))
  _logger.debug('reuse chains: %s', chains)

  for idx, chain in enumerate(chains):
    _logger.debug('reuse chain %d of tensor %s: %s', idx, tensor.name, chain)
  return chains

def _get_points(tile_size, tensor, unroll_factor):
  """Generates offset-to-point mapping for a Tensor.

  Generates a mapping which can be used to determine the accessed point index
  from the offset for a Tensor, under the given tile size and unroll factor.

  Args:
    tile_size: An iterable representing the tile size in each dimension.
    tensor: A Tensor to which the mapping belongs.
    unroll_factor: An int representing the unroll factor.

  Returns:
    A dict of name str to a dict of offset to a dict of unroll index to
    point index.
  """

  all_points = {} # {name: {offset: {unroll_idx: point_idx}}}
  for child in tensor.children.values():
    all_points[child.name] = {}
    offsets = child.ld_offsets[tensor.name]
    for unroll_idx in range(unroll_factor):
      for idx, offset in enumerate(offsets):
        all_points[child.name].setdefault(
          max(offsets) - offset + child.ld_delays[tensor.name] + unroll_idx,
          {})[unroll_factor-1-unroll_idx] = idx
  for child in tensor.children.values():
    for offset, points in all_points[child.name].items():
      for unroll_idx, point in points.items():
        _logger.debug(
          '%s <- %s @ offset=%d <=> %s @ unroll_idx=%d',
          child.name, tensor.name, offset,
          util.idx2str(list(child.ld_indices[tensor.name].values())[point].idx),
          unroll_idx)
  return all_points

def _get_reuse_buffer(tile_size, tensor, unroll_factor):
  """Generates reuse buffer for a Tensor.

  Generates a list representing the reuse buffer for a Tensor, under the given
  tile size and unroll factor.

  Args:
    tile_size: An iterable representing the tile size in each dimension.
    tensor: A Tensor to which the mapping belongs.
    unroll_factor: An int representing the unroll factor.

  Returns:
    A list whose first element is an int representing the length of the
    reuse buffer (capacity of data element), followed by unroll_factor
    number of (start, end) tuples, where start and end are the offsets from
    the lastest input of each piece of the reuse buffer.
  """

  reuse_buffer = [None]  # [length, (start, end), (start, end), ...]
  offsets = []
  for chain in _get_reuse_chains(tile_size, tensor, unroll_factor):
    reuse_buffer.append((chain[0], chain[0]))
    offsets.append(chain[0])
    for j in range(len(chain)-1):
      reuse_buffer.append((chain[j], chain[j+1]))
      offsets.append(chain[j+1])
  reuse_buffer[0] = max(offsets)+1
  _logger.debug('reuse chains of tensor %s: %s' % (tensor.name, reuse_buffer))
  return reuse_buffer

def get_indices_id(indices):
  return '_'.join(str(idx).replace('-', 'm') for idx in indices)

def get_stencil_distance(stencil_window, tile_size):
  return (max(soda_util.serialize_iter(stencil_window, tile_size)) +
          soda_util.serialize(get_stencil_window_offset(stencil_window),
                              tile_size))

def get_stencil_dim(points):
  dimension = len(next(iter(points)))
  return [max_index-min_index+1 for max_index, min_index in zip(
    [max([point[dim] for point in points]) for dim in range(dimension)],
    [min([point[dim] for point in points]) for dim in range(dimension)])]

_overall_stencil_window_cache = {}
def get_overall_stencil_window(input_tensor, output_tensor):
  if isinstance(input_tensor, collections.Iterable):
    all_points = tuple(sorted(set.union(*(
        set(get_overall_stencil_window(_, output_tensor))
        for _ in input_tensor))))
    _logger.debug(
        'overall stencil window of %s (%s) <- {%s} is %s (%d points)',
        output_tensor.name, ', '.join(['0']*len(output_tensor.st_idx)),
        ', '.join(_.name for _ in input_tensor), all_points, len(all_points))
    return all_points
  # normalize store index to 0
  idx = (id(input_tensor), id(output_tensor))
  if idx in _overall_stencil_window_cache:
    return _overall_stencil_window_cache[idx]
  _logger.debug('get overall stencil window of %s <- %s',
                output_tensor.name, input_tensor.name)
  all_points = set()
  for name, points in output_tensor.ld_indices.items():
    _logger.debug('%s@%s <- %s', output_tensor.name,
                  util.idx2str(output_tensor.st_idx),
                  util.idx2str(points.values()))
    if name != input_tensor.name:
      recursive_points = get_overall_stencil_window(
          input_tensor, output_tensor.parents[name])
      _logger.debug('recursive points: %s', util.idx2str(recursive_points))
      all_points |= set.union(*[{
          tuple(map(lambda a, b, c: a + b - c, _, point, output_tensor.st_idx))
          for _ in recursive_points} for point in points])
    else:
      all_points |= {tuple(map(operator.sub, point, output_tensor.st_idx))
                     for point in points}
  all_points = tuple(sorted(all_points))
  _logger.debug('overall stencil window of %s (%s) <- %s is %s (%d points)',
                output_tensor.name, ', '.join(['0']*len(output_tensor.st_idx)),
                input_tensor.name, all_points, len(all_points))
  _overall_stencil_window_cache[idx] = all_points
  return all_points

def get_stencil_window_offset(stencil_window):
  # only works if window is normalized to store at 0
  return tuple(-min(p[d] for p in stencil_window)
         for d in range(len(next(iter(stencil_window)))))
