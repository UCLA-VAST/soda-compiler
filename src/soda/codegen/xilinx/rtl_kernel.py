import collections
import concurrent
import logging
import os
import shutil
import sys
import tarfile
import tempfile

from haoda import util
from haoda.backend import xilinx as backend
from soda.codegen.xilinx import hls_kernel

_logger = logging.getLogger().getChild(__name__)

def print_code(stencil, xo_file, platform=None, jobs=os.cpu_count()):
  """Generate hardware object file for the given Stencil.

  Working `vivado` and `vivado_hls` is required in the PATH.

  Args:
    stencil: Stencil object to generate from.
    xo_file: file object to write to.
    platform: path to the SDAccel platform directory.
    jobs: maximum number of jobs running in parallel.
  """

  m_axi_names = []
  m_axi_bundles = []
  inputs = []
  outputs = []
  for stmt in stencil.output_stmts + stencil.input_stmts:
    for bank in stmt.dram:
      haoda_type = 'uint%d' % stencil.burst_width
      bundle_name = util.get_bundle_name(stmt.name, bank)
      m_axi_names.append(bundle_name)
      m_axi_bundles.append((bundle_name, haoda_type))

  for stmt in stencil.output_stmts:
    for bank in stmt.dram:
      haoda_type = 'uint%d' % stencil.burst_width
      bundle_name = util.get_bundle_name(stmt.name, bank)
      outputs.append((util.get_port_name(stmt.name, bank), bundle_name,
                      haoda_type, util.get_port_buf_name(stmt.name, bank)))
  for stmt in stencil.input_stmts:
    for bank in stmt.dram:
      haoda_type = 'uint%d' % stencil.burst_width
      bundle_name = util.get_bundle_name(stmt.name, bank)
      inputs.append((util.get_port_name(stmt.name, bank), bundle_name,
                     haoda_type, util.get_port_buf_name(stmt.name, bank)))

  top_name = stencil.app_name + '_kernel'

  if 'XDEVICE' in os.environ:
    xdevice = os.environ['XDEVICE'].replace(':', '_').replace('.', '_')
    if platform is None or not os.path.exists(platform):
      platform = os.path.join('/opt/xilinx/platforms', xdevice)
    if platform is None or not os.path.exists(platform):
      if 'XILINX_SDX' in os.environ:
        platform = os.path.join(os.environ['XILINX_SDX'], 'platforms', xdevice)
  if platform is None or not os.path.exists(platform):
    raise ValueError('Cannot determine platform from environment.')
  device_info = backend.get_device_info(platform)

  with tempfile.TemporaryDirectory(prefix='sodac-xrtl-') as tmpdir:
    dataflow_kernel = os.path.join(tmpdir, 'dataflow_kernel.cpp')
    with open(dataflow_kernel, 'w') as dataflow_kernel_obj:
      print_dataflow_hls_interface(
          util.Printer(dataflow_kernel_obj), top_name, inputs, outputs)

    kernel_xml = os.path.join(tmpdir, 'kernel.xml')
    with open(kernel_xml, 'w') as kernel_xml_obj:
      backend.print_kernel_xml(top_name, outputs + inputs, kernel_xml_obj)

    kernel_file = os.path.join(tmpdir, 'kernel.cpp')
    with open(kernel_file, 'w') as kernel_fileobj:
      hls_kernel.print_code(stencil, kernel_fileobj)

    super_source = stencil.dataflow_super_source
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
      threads = []
      for module_id in range(len(super_source.module_traits)):
        threads.append(executor.submit(
            synthesis_module, tmpdir, [kernel_file],
            util.get_func_name(module_id), device_info))
      threads.append(executor.submit(
          synthesis_module, tmpdir, [dataflow_kernel], top_name, device_info))
      for future in concurrent.futures.as_completed(threads):
        returncode, stdout, stderr = future.result()
        log_func = _logger.error if returncode != 0 else _logger.debug
        if stdout:
          log_func(stdout.decode())
        if stderr:
          log_func(stderr.decode())
        if returncode != 0:
          util.pause_for_debugging()
          sys.exit(returncode)

    hdl_dir = os.path.join(tmpdir, 'hdl')
    with open(os.path.join(hdl_dir, 'Dataflow.v'), mode='w') as dataflow_v:
      print_top_module(backend.VerilogPrinter(dataflow_v),
                       stencil.dataflow_super_source, inputs, outputs)

    util.pause_for_debugging()

    xo_filename = os.path.join(tmpdir, stencil.app_name + '.xo')
    with backend.PackageXo(xo_filename, top_name, kernel_xml, hdl_dir,
                           m_axi_names, [dataflow_kernel]) as proc:
      stdout, stderr = proc.communicate()
    log_func = _logger.error if proc.returncode != 0 else _logger.debug
    log_func(stdout.decode())
    log_func(stderr.decode())
    with open(xo_filename, mode='rb') as xo_fileobj:
      shutil.copyfileobj(xo_fileobj, xo_file)

def synthesis_module(tmpdir, kernel_files, module_name, device_info):
  """Synthesis a module in kernel files.

  Returns:
    (returncode, stdout, stderr) results of the subprocess.
  """
  with tempfile.TemporaryFile(mode='w+b') as tarfileobj:
    with backend.RunHls(
        tarfileobj, kernel_files, module_name, device_info['clock_period'],
        device_info['part_num']) as proc:
      stdout, stderr = proc.communicate()
    if proc.returncode == 0:
      tarfileobj.seek(0)
      with tarfile.open(mode='r', fileobj=tarfileobj) as tar:
        tar.extractall(tmpdir, filter(lambda _: _.name.startswith('hdl'),
                                      tar.getmembers()))
  return proc.returncode, stdout, stderr

FIFO_PORT_SUFFIXES = dict(
    data_in='_din',
    not_full='_full_n',
    write_enable='_write',
    data_out='_dout',
    not_empty='_empty_n',
    read_enable='_read',
    not_block='_blk_n')


def print_top_module(printer, super_source, inputs, outputs):
  println = printer.println
  println('`timescale 1 ns / 1 ps')
  args = ['ap_clk', 'ap_rst', 'ap_start', 'ap_done', 'ap_continue', 'ap_idle',
          'ap_ready']
  for port_name, _, _, _ in outputs:
    args.append('{}_V_V{data_in}'.format(port_name, **FIFO_PORT_SUFFIXES))
    args.append('{}_V_V{not_full}'.format(port_name, **FIFO_PORT_SUFFIXES))
    args.append('{}_V_V{write_enable}'.format(port_name, **FIFO_PORT_SUFFIXES))
  for port_name, _, _, _ in inputs:
    args.append('{}_V_V{data_out}'.format(port_name, **FIFO_PORT_SUFFIXES))
    args.append('{}_V_V{not_empty}'.format(port_name, **FIFO_PORT_SUFFIXES))
    args.append('{}_V_V{read_enable}'.format(port_name, **FIFO_PORT_SUFFIXES))
  printer.module('Dataflow', args)
  println()

  input_args = 'ap_clk', 'ap_rst', 'ap_start', 'ap_continue'
  output_args = 'ap_done', 'ap_idle', 'ap_ready'

  for arg in input_args:
    println('input  %s;' % arg)
  for arg in output_args:
    println('output %s;' % arg)
  for port_name, _, haoda_type, _ in outputs:
    kwargs = dict(port_name=port_name, **FIFO_PORT_SUFFIXES)
    println('output [{}:0] {port_name}_V_V{data_in};'.format(
        util.get_width_in_bits(haoda_type) - 1, **kwargs))
    println('input  {port_name}_V_V{not_full};'.format(**kwargs))
    println('output {port_name}_V_V{write_enable};'.format(**kwargs))
  for port_name, _, haoda_type, _ in inputs:
    kwargs = dict(port_name=port_name, **FIFO_PORT_SUFFIXES)
    println('input  [{}:0] {port_name}_V_V{data_out};'.format(
        util.get_width_in_bits(haoda_type) - 1, **kwargs))
    println('input  {port_name}_V_V{not_empty};'.format(**kwargs))
    println('output {port_name}_V_V{read_enable};'.format(**kwargs))
  println()

  println("reg ap_done = 1'b0;")
  println("reg ap_idle = 1'b1;")
  println("reg ap_ready = 1'b0;")

  for port_name, _, haoda_type, _ in outputs:
    kwargs = dict(port_name=port_name, **FIFO_PORT_SUFFIXES)
    println('reg [{}:0] {port_name}{data_in};'.format(
        util.get_width_in_bits(haoda_type) - 1, **kwargs))
    println('wire {port_name}_V_V{write_enable};'.format(**kwargs))
  for port_name, _, haoda_type, _ in inputs:
    println('wire {}_V_V{read_enable};'.format(port_name, **FIFO_PORT_SUFFIXES))
  println('reg ap_rst_n_inv;')
  with printer.always('*'):
    println('ap_rst_n_inv = ap_rst;')
  println()

  with printer.always('posedge ap_clk'):
    with printer.if_('ap_rst'):
      println("ap_done <= 1'b0;")
      println("ap_idle <= 1'b1;")
      println("ap_ready <= 1'b0;")
      printer.else_()
      println('ap_idle <= ~ap_start;')

  for port_name, _, _, _ in outputs:
    println('reg {}_V_V{not_block};'.format(port_name, **FIFO_PORT_SUFFIXES))
  for port_name, _, _, _ in inputs:
    println('reg {}_V_V{not_block};'.format(port_name, **FIFO_PORT_SUFFIXES))

  with printer.always('*'):
    for port_name, _, _, _ in outputs:
      println('{port_name}_V_V{not_block} = {port_name}_V_V{not_full};'.format(
          port_name=port_name, **FIFO_PORT_SUFFIXES))
    for port_name, _, _, _ in inputs:
      println('{port_name}_V_V{not_block} = {port_name}_V_V{not_empty};'.format(
          port_name=port_name, **FIFO_PORT_SUFFIXES))
  println()

  for module in super_source.tpo_node_gen():
    for fifo in module.fifos:
      kwargs = {
          'name' : fifo.c_expr,
          'msb' : fifo.width_in_bits - 1,
          **FIFO_PORT_SUFFIXES
      }
      println('wire [{msb}:0] {name}{data_in};'.format(**kwargs))
      println('wire {name}{not_full};'.format(**kwargs))
      println('wire {name}{write_enable};'.format(**kwargs))
      println('wire [{msb}:0] {name}{data_out};'.format(**kwargs))
      println('wire {name}{not_empty};'.format(**kwargs))
      println('wire {name}{read_enable};'.format(**kwargs))
      println()

      args = collections.OrderedDict((
          ('clk', 'ap_clk'),
          ('reset', 'ap_rst_n_inv'),
          ('if_read_ce', "1'b1"),
          ('if_write_ce', "1'b1"),
          ('if{data_in}'.format(**kwargs),
           '{name}{data_in}'.format(**kwargs)),
          ('if{not_full}'.format(**kwargs),
           '{name}{not_full}'.format(**kwargs)),
          ('if{write_enable}'.format(**kwargs),
           '{name}{write_enable}'.format(**kwargs)),
          ('if{data_out}'.format(**kwargs),
           '{name}{data_out}'.format(**kwargs)),
          ('if{not_empty}'.format(**kwargs),
           '{name}{not_empty}'.format(**kwargs)),
          ('if{read_enable}'.format(**kwargs),
           '{name}{read_enable}'.format(**kwargs))
      ))
      printer.module_instance('fifo_w{width}_d{depth}_A'.format(
          width=fifo.width_in_bits, depth=fifo.depth+2), fifo.c_expr, args)
      println()

  for module in super_source.tpo_node_gen():
    module_trait, module_trait_id = super_source.module_table[module]
    args = collections.OrderedDict((('ap_clk', 'ap_clk'),
                        ('ap_rst', 'ap_rst_n_inv'),
                        ('ap_start', "1'b1")))
    for dram_ref, bank in module.dram_writes:
      kwargs = dict(port=dram_ref.dram_fifo_name(bank),
                    fifo=util.get_port_name(dram_ref.var, bank),
                    **FIFO_PORT_SUFFIXES)
      args['{port}_V{data_in}'.format(**kwargs)] = \
                       '{fifo}_V_V{data_in}'.format(**kwargs)
      args['{port}_V{not_full}'.format(**kwargs)] = \
                       '{fifo}_V_V{not_full}'.format(**kwargs)
      args['{port}_V{write_enable}'.format(**kwargs)] = \
                       '{fifo}_V_V{write_enable}'.format(**kwargs)
    for port, fifo in zip(module_trait.output_fifos, module.output_fifos):
      kwargs = dict(port=port, fifo=fifo, **FIFO_PORT_SUFFIXES)
      args['{port}_V{data_in}'.format(**kwargs)] = \
                       '{fifo}{data_in}'.format(**kwargs)
      args['{port}_V{not_full}'.format(**kwargs)] = \
                       '{fifo}{not_full}'.format(**kwargs)
      args['{port}_V{write_enable}'.format(**kwargs)] = \
                       '{fifo}{write_enable}'.format(**kwargs)
    for port, fifo in zip(module_trait.input_fifos, module.input_fifos):
      kwargs = dict(port=port, fifo=fifo, **FIFO_PORT_SUFFIXES)
      args['{port}_V{data_out}'.format(**kwargs)] = \
                       "{{1'b1, {fifo}{data_out}}}".format(**kwargs)
      args['{port}_V{not_empty}'.format(**kwargs)] = \
                       '{fifo}{not_empty}'.format(**kwargs)
      args['{port}_V{read_enable}'.format(**kwargs)] = \
                       '{fifo}{read_enable}'.format(**kwargs)
    for dram_ref, bank in module.dram_reads:
      kwargs = dict(port=dram_ref.dram_fifo_name(bank),
                    fifo=util.get_port_name(dram_ref.var, bank),
                    **FIFO_PORT_SUFFIXES)
      args['{port}_V{data_out}'.format(**kwargs)] = \
                       "{{1'b1, {fifo}_V_V{data_out}}}".format(**kwargs)
      args['{port}_V{not_empty}'.format(**kwargs)] = \
                       '{fifo}_V_V{not_empty}'.format(**kwargs)
      args['{port}_V{read_enable}'.format(**kwargs)] = \
                       '{fifo}_V_V{read_enable}'.format(**kwargs)
    printer.module_instance(util.get_func_name(module_trait_id), module.name,
                            args)
    println()
  printer.endmodule()

  fifos = set()
  for module in super_source.tpo_node_gen():
    for fifo in module.fifos:
      fifos.add((fifo.width_in_bits, fifo.depth + 2))
  for fifo in fifos:
    printer.fifo_module(*fifo)

def print_dataflow_hls_interface(printer, top_name, inputs, outputs):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  do_indent = printer.do_indent
  un_indent = printer.un_indent
  m_axi_ports = outputs + inputs
  print_func = printer.print_func

  println('#include <cstddef>')
  println('#include <cstdint>')
  println('#include <ap_int.h>')
  println('#include <hls_stream.h>')

  println('template<int kBurstWidth>')
  print_func('void BurstRead', [
      'hls::stream<ap_uint<kBurstWidth>>* to', 'ap_uint<kBurstWidth>* from',
      'uint64_t data_num'], align=0)
  do_scope()
  println('load_epoch:', 0)
  with printer.for_('uint64_t epoch = 0', 'epoch < data_num', '++epoch'):
    println('#pragma HLS pipeline II=1', 0)
    println('to->write(from[epoch]);')
  un_scope()

  println('template<int kBurstWidth>')
  print_func('void BurstWrite', [
      'ap_uint<kBurstWidth>* to', 'hls::stream<ap_uint<kBurstWidth>>* from',
      'uint64_t data_num'], align=0)
  do_scope()
  println('store_epoch:', 0)
  with printer.for_('uint64_t epoch = 0', 'epoch < data_num', '++epoch'):
    println('#pragma HLS pipeline II=1', 0)
    println('to[epoch] = from->read();')
  un_scope()

  params = ['hls::stream<{}>* {}'.format(util.get_c_type(haoda_type), name)
            for name, _, haoda_type, _ in m_axi_ports]
  print_func('void Dataflow', params, align=0)
  do_scope()
  for name, _, haoda_type, _ in inputs:
    println('volatile {c_type} {name}_read;'.format(
        c_type=util.get_c_type(haoda_type), name=name))
  for name, _, haoda_type, _ in inputs:
    println('{name}_read = {name}->read();'.format(name=name))
  for name, _, haoda_type, _ in outputs:
    println(
        '{name}->write({c_type}());'.format(
        c_type=util.get_c_type(haoda_type), name=name))
  un_scope()

  params = ['{}* {}'.format(util.get_c_type(haoda_type), name)
            for name, _, haoda_type, _ in m_axi_ports]
  params.append('uint64_t coalesced_data_num')
  print_func('void %s' % top_name, params, align=0)
  do_scope()

  println('#pragma HLS dataflow', 0)

  for port_name, bundle_name, _, _ in m_axi_ports:
    println('#pragma HLS interface m_axi port={} offset=slave bundle={}'.format(
        port_name, bundle_name), 0)
  for port_name, _, _, _ in m_axi_ports:
    println('#pragma HLS interface s_axilite port={} bundle=control'.format(
        port_name), 0)
  println('#pragma HLS interface s_axilite port=coalesced_data_num '
          'bundle=control', 0)
  println('#pragma HLS interface s_axilite port=return bundle=control', 0)
  println()

  for _, _, haoda_type, name in m_axi_ports:
    println('hls::stream<{c_type}> {name}("{name}");'.format(
        c_type=util.get_c_type(haoda_type), name=name))
    println('#pragma HLS stream variable={name} depth=32'.format(name=name), 0)

  for port_name, _, haoda_type, buf_name in inputs:
    print_func('BurstRead', [
        '&{name}'.format(name=buf_name), '{name}'.format(name=port_name),
        'coalesced_data_num'], suffix=';', align=0)

  params = ['&{}'.format(name) for _, _, _, name in m_axi_ports]
  printer.print_func('Dataflow', params, suffix=';', align=0)

  for port_name, _, haoda_type, buf_name in outputs:
    print_func('BurstWrite', [
        '{name}'.format(name=port_name), '&{name}'.format(name=buf_name),
        'coalesced_data_num'],
               suffix=';', align=0)

  un_scope()
