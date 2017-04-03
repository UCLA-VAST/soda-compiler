import logging

from haoda import ir
from soda import util

_logger = logging.getLogger().getChild(__name__)

GRAMMAR = r'''
SodaProgram:
(
  ('burst' 'width' ':' burst_width=INT)
  ('iterate' ':' iterate=INT)
  ('kernel' ':' app_name=ID)
  ('unroll' 'factor' ':' unroll_factor=INT)
  (input_stmts=InputStmt)+
  (param_stmts=ParamStmt)*
  (local_stmts=LocalStmt)*
  (output_stmts=OutputStmt)+
)#;

YesOrNo: 'yes'|'no';

Comment: /\s*#.*$/;

FuncName: 'cos'|'sin'|'tan'|'acos'|'asin'|'atan'|'atan2'|
  'cosh'|'sinh'|'tanh'|'acosh'|'asinh'|'atanh'|
  'exp'|'frexp'|'ldexp'|'log'|'log10'|'modf'|'exp2'|'expm1'|'ilogb'|'log1p'|'log2'|'logb'|'scalbn'|'scalbln'|
  'pow'|'sqrt'|'cbrt'|'hypot'|
  'erf'|'erfc'|'tgamma'|'lgamma'|
  'ceil'|'floor'|'fmod'|'trunc'|'round'|'lround'|'llround'|'rint'|'lrint'|'llrint'|'nearbyint'|'remainder'|'remquo'|
  'copysign'|'nan'|'nextafter'|'nexttoward'|'fdim'|'fmax'|'fmin'|'fabs'|'abs'|'fma'|
  'min'|'max'|'select';

InputStmt: 'input' ('dram' dram=INT ('.' dram=INT)*)? haoda_type=Type ':' name=ID ('(' (tile_size=INT ',')* '*' ')')?;
LocalStmt: 'local' haoda_type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;
OutputStmt: 'output' ('dram' dram=INT ('.' dram=INT)*)? haoda_type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;

ParamStmt: 'param' ('dram' dram=INT ('.' dram=INT)*)? haoda_type=Type (',' attr=ParamAttr)* ':' name=ID ('[' size=INT ']')*;
ParamAttr: 'dup' dup=Int | partitioning=Partitioning;
Partitioning:
  'partition' strategy='complete' ('dim' '=' dim=Int)? |
  'partition' strategy='cyclic' 'factor' '=' factor=Int ('dim' '=' dim=Int)?;
''' + ir.GRAMMAR

class InputStmt(ir.Node):
  """Node for input statement, represents a tiled input tensor.

  Attributes:
    haoda_type: Type of this input tensor.
    dram: [int], dram id used to read this input
    name: str, name of this input tensor.
    tile_size: list of tile sizes. The last dimension should be 0.
  """
  SCALAR_ATTRS = 'haoda_type', 'name'
  LINEAR_ATTRS = ('tile_size', 'dram',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # pylint: disable=access-member-before-definition
    if not self.dram:
      self.dram = (0,)
    self.tile_size += (0,)

  def __str__(self):
    result = 'input {}: {}'.format(self.haoda_type, self.name)
    if self.tile_size[:-1]:
      result += '({}, *)'.format(', '.join(map(str, self.tile_size[:-1])))
    return result

class LocalStmtOrOutputStmt(ir.Node):
  SCALAR_ATTRS = 'haoda_type', 'ref', 'expr'
  LINEAR_ATTRS = ('let',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    var_types = {}
    # pylint: disable=access-member-before-definition
    for let in self.let:
      var_types[let.name] = let.haoda_type
    def set_var_type(obj, var_types):
      if isinstance(obj, ir.Var) and obj.name in var_types:
        obj.haoda_type = var_types[obj.name]
      return obj
    self.let = tuple(_.visit(set_var_type, var_types) for _ in self.let)
    self.expr = self.expr.visit(set_var_type, var_types)

  @property
  def name(self):
    return self.ref.name

  def __str__(self):
    if self.let:
      let = '\n  {}\n '.format('\n  '.join(map(str, self.let)))
    else:
      let = ''
    return '{} {}:{} {} = {}'.format(type(self).__name__[:-4].lower(),
                                     self.haoda_type, let, self.ref,
                                     ir.unparenthesize(self.expr))

class LocalStmt(LocalStmtOrOutputStmt):
  pass

class OutputStmt(LocalStmtOrOutputStmt):
  LINEAR_ATTRS = LocalStmtOrOutputStmt.LINEAR_ATTRS + ('dram',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # pylint: disable=access-member-before-definition
    if not self.dram:
      self.dram = (0,)

class ParamStmt(ir.Node):
  SCALAR_ATTRS = 'haoda_type', 'attr', 'name', 'size'
  LINEAR_ATTRS = ('dram',)
  def __str__(self):
    return 'param {}{}: {}{}'.format(
      self.haoda_type, ''.join(map(', {}'.format, self.attr)),
      self.name, ''.join(map('[{}]'.format, self.size)))

class ParamAttr(ir.Node):
  SCALAR_ATTRS = 'dup', 'partitioning'
  def __str__(self):
    if self.dup is not None:
      return 'dup {}'.format(self.dup)
    result = 'partition {0.strategy}'.format(self.partitioning)
    if self.partitioning.strategy == 'cyclic':
      result += ' factor={}'.format(self.partitioning.factor)
    if self.partitioning.dim is not None:
      result += ' dim={}'.format(self.partitioning.dim)
    return result

class SodaProgram(ir.Node):
  SCALAR_ATTRS = ('burst_width', 'iterate', 'app_name', 'unroll_factor',
                  'input_stmts', 'param_stmts', 'local_stmts', 'output_stmts')
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    for node in self.input_stmts:
      if hasattr(self, 'tile_size'):
        # pylint: disable=access-member-before-definition
        if self.tile_size != node.tile_size:
          msg = ('tile size %s doesn\'t match previous one %s' %
               # pylint: disable=access-member-before-definition
               (node.tile_size, self.tile_size))
          raise util.SemanticError(msg)
      elif node.tile_size[:-1]:
        self.tile_size = node.tile_size
        self.dim = len(self.tile_size)
    # deal with 1D case
    if not hasattr(self, 'tile_size'):
      # pylint: disable=undefined-loop-variable
      self.tile_size = node.tile_size
      self.dim = len(self.tile_size)

  def __str__(self):
    return '\n'.join(filter(None, (
      'burst width: {}'.format(self.burst_width),
      'iterate: {}'.format(self.iterate),
      'kernel: {}'.format(self.app_name),
      'unroll factor: {}'.format(self.unroll_factor),
      '\n'.join(map(str, self.input_stmts)),
      '\n'.join(map(str, self.param_stmts)),
      '\n'.join(map(str, self.local_stmts)),
      '\n'.join(map(str, self.output_stmts)))))

CLASSES = (
  InputStmt,
  LocalStmt,
  OutputStmt,
  ir.Let,
  ir.Ref,
  ir.Expr,
  ir.LogicAnd,
  ir.BinaryOr,
  ir.Xor,
  ir.BinaryAnd,
  ir.EqCmp,
  ir.LtCmp,
  ir.AddSub,
  ir.MulDiv,
  ir.Unary,
  ir.Operand,
  ir.Cast,
  ir.Call,
  ir.Var,
  ParamStmt,
  ParamAttr,
  SodaProgram,
)
