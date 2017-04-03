import logging

from haoda import util

logger = logging.getLogger().getChild(__name__)

def print_code(stencil, header_file):
  logger.info('generate host header code as %s' % header_file.name)
  printer = util.Printer(header_file)
  println = printer.println
  do_indent = printer.do_indent
  un_indent = printer.un_indent
  println('#ifndef HALIDE_%s_H_' % stencil.app_name.upper())
  println('#define HALIDE_%s_H_' % stencil.app_name.upper())
  println()

  println('#ifndef HALIDE_ATTRIBUTE_ALIGN')
  do_indent()
  println('#ifdef _MSC_VER')
  do_indent()
  println('#define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))')
  un_indent()
  println('#else')
  do_indent()
  println('#define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))')
  un_indent()
  println('#endif')
  un_indent()
  println('#endif//HALIDE_ATTRIBUTE_ALIGN')
  println()

  println('#ifndef BUFFER_T_DEFINED')
  println('#define BUFFER_T_DEFINED')
  println('#include<stdbool.h>')
  println('#include<stdint.h>')
  println('typedef struct buffer_t {')
  do_indent()
  println('uint64_t dev;')
  println('uint8_t* host;')
  println('int32_t extent[4];')
  println('int32_t stride[4];')
  println('int32_t min[4];')
  println('int32_t elem_size;')
  println('HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;')
  println('HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;')
  println('HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];')
  un_indent()
  println('} buffer_t;')
  println('#endif//BUFFER_T_DEFINED')
  println()

  println('#ifndef HALIDE_FUNCTION_ATTRS')
  println('#define HALIDE_FUNCTION_ATTRS')
  println('#endif//HALIDE_FUNCTION_ATTRS')
  println()

  tensors = stencil.input_names + stencil.output_names + stencil.param_names
  println('int {}({}const char* xclbin) HALIDE_FUNCTION_ATTRS;'.format(
      stencil.app_name,
      ''.join(map('buffer_t *var_{}_buffer, '.format, tensors))))
  println()

  println('#endif//HALIDE_%s_H_' % stencil.app_name.upper())
  println()
