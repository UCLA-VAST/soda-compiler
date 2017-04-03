import contextlib
import logging
import signal

# constants
COORDS_TILED = 'xyzw'
COORDS_IN_TILE = 'ijkl'
COORDS_IN_ORIG = 'pqrs'
TYPE_WIDTH = {
  'float': 32,
  'double': 64,
  'half': 16
}
MAX_DRAM_BANK = 4

_logger = logging.getLogger().getChild(__name__)

class InternalError(Exception):
  pass

class SemanticError(Exception):
  pass

class SemanticWarn(Exception):
  pass

class Printer():
  def __init__(self, out):
    self._out = out
    self._indent = 0
    self._assign = 0
    self._comments = []
    self._tab = 2

  def println(self, line='', indent=-1):
    if indent < 0:
      indent = self._indent
    if line:
      self._out.write('%s%s\n' % (' '*indent*self._tab, line))
    else:
      self._out.write('\n')

  def do_indent(self):
    self._indent += 1

  def un_indent(self):
    self._indent -= 1

  def do_scope(self, comment=''):
    self.println('{')
    self.do_indent()
    self._comments.append(comment)

  def un_scope(self, comment='', suffix=''):
    self.un_indent()
    popped_comment = self._comments.pop()
    if comment:
      self.println('}%s // %s' % (suffix, comment))
    else:
      if popped_comment:
        self.println('}%s // %s' % (suffix, popped_comment))
      else:
        self.println('}%s' % suffix)

  def new_var(self):
    self._assign += 1
    return self.last_var()

  def last_var(self, offset=-1):
    return 'assign_%d' % (self._assign+offset)

  def print_func(self, name, params, suffix='', align=80):
    lines = [name+'(']
    for param in params:
      if ((self._indent + min(1, len(lines)-1))*self._tab+
          len(lines[-1])+len(param+', ')) > align:
        lines.append(param+', ')
      else:
        lines[-1] += param+', '
    if lines[-1][-2:] == ', ':
      lines[-1] = lines[-1][:-2]+')'+suffix
    line = lines.pop(0)
    self.println(line)
    if lines:
      self.do_indent()
      for line in lines:
        self.println(line)
      self.un_indent()

  @contextlib.contextmanager
  def for_(self, *args):
    if len(args) == 3:
      self.println('for ({}; {}; {}) {{'.format(*args))
    elif len(args) == 2:
      self.println('for ({} : {}) {{'.format(*args))
    else:
      raise InternalError('for_ takes 2 or 3 arguments')
    self.do_indent()
    yield
    self.un_indent()
    self.println('}')

  @contextlib.contextmanager
  def do_while(self, cond):
    self.println('do {')
    self.do_indent()
    yield
    self.un_indent()
    self.println('}} while ({});'.format(cond))

  @contextlib.contextmanager
  def if_(self, cond):
    self.println('if ({}) {{'.format(cond))
    self.do_indent()
    yield
    self.un_indent()
    self.println('}')

  @contextlib.contextmanager
  def elif_(self, cond):
    self.un_indent()
    self.println('}} else if ({}) {{'.format(cond))
    self.do_indent()
    yield

  @contextlib.contextmanager
  def else_(self):
    self.un_indent()
    self.println('} else {')
    self.do_indent()
    yield

def print_define(printer, var, val):
  printer.println('#ifndef %s' % var)
  printer.println('#define %s %d' % (var, val))
  printer.println('#endif//%s' % var)

def print_guard(printer, var, val):
  printer.println('#ifdef %s' % var)
  printer.println('#if %s != %d' % (var, val))
  printer.println('#error %s != %d' % (var, val))
  printer.println('#endif//%s != %d' % (var, val))
  printer.println('#endif//%s' % var)

def get_c_type(haoda_type):
  if haoda_type in {
      'uint8', 'uint16', 'uint32', 'uint64',
      'int8', 'int16', 'int32', 'int64'}:
    return haoda_type+'_t'
  if haoda_type is None:
    return None
  if haoda_type == 'float32':
    return 'float'
  if haoda_type == 'float64':
    return 'double'
  for token in ('int', 'uint'):
    if haoda_type.startswith(token):
      return 'ap_{}<{}>'.format(token, haoda_type.replace(token, ''))
  return haoda_type

def get_haoda_type(c_type):
  return c_type[:-2] if c_type[-2:] == '_t' else c_type

def get_width_in_bits(haoda_type):
  if isinstance(haoda_type, str):
    if haoda_type in TYPE_WIDTH:
      return TYPE_WIDTH[haoda_type]
    for prefix in 'uint', 'int', 'float':
      if haoda_type.startswith(prefix):
        return int(haoda_type.lstrip(prefix).split('_')[0])
  else:
    if hasattr(haoda_type, 'haoda_type'):
      return get_width_in_bits(haoda_type.haoda_type)
  raise InternalError('unknown haoda type: %s' % haoda_type)

def get_width_in_bytes(haoda_type):
  return (get_width_in_bits(haoda_type)-1)//8+1

def is_float(haoda_type):
  return haoda_type in {'half', 'double'} or haoda_type.startswith('float')

def idx2str(idx):
  return '(%s)' % ', '.join(map(str, idx))

def lst2str(idx):
  return '[%s]' % ', '.join(map(str, idx))

def get_module_name(module_id):
  return 'module_%d' % module_id

def get_func_name(module_id):
  return 'Module%dFunc' % module_id

get_port_name = lambda name, bank: 'bank_{}_{}'.format(bank, name)
get_port_buf_name = lambda name, bank: 'bank_{}_{}_buf'.format(bank, name)
def get_bundle_name(name, bank):
  return '{}_bank_{}'.format(name.replace('<', '_').replace('>', ''), bank)

def pause_for_debugging():
  if _logger.isEnabledFor(logging.DEBUG):
    try:
      _logger.debug('pausing for debugging... send Ctrl-C to resume')
      signal.pause()
    except KeyboardInterrupt:
      pass
