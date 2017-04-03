import contextlib
import collections
import logging
import os
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from haoda import util

_logger = logging.getLogger().getChild(__name__)

class Vivado(subprocess.Popen):
  """Call vivado with the given tcl commands and arguments.

  Args:
    commands: string of tcl commands
    args: sequence of arguments
  """
  def __init__(self, commands, *args):
    self.cwd = tempfile.TemporaryDirectory(prefix='vivado-')
    self.tcl_file = open(os.path.join(self.cwd.name, 'tcl'), mode='w+')
    self.tcl_file.write(commands)
    self.tcl_file.flush()
    cmd_args = ['vivado', '-mode', 'batch', '-source', self.tcl_file.name,
                '-nojournal', '-nolog', '-tclargs', *args]
    pipe_args = {'stdout' : subprocess.PIPE, 'stderr' : subprocess.PIPE}
    super().__init__(cmd_args, cwd=self.cwd.name, **pipe_args)

  def __exit__(self, *args):
    super().__exit__(*args)
    self.tcl_file.close()
    self.cwd.cleanup()

class VivadoHls(subprocess.Popen):
  """Call vivado_hls with the given tcl commands.

  Args:
    commands: string of tcl commands
  """
  def __init__(self, commands):
    self.cwd = tempfile.TemporaryDirectory(prefix='vivado-hls-')
    self.tcl_file = open(os.path.join(self.cwd.name, 'tcl'), mode='w+')
    self.tcl_file.write(commands)
    self.tcl_file.flush()
    cmd_args = ['vivado_hls', '-f', self.tcl_file.name, '-l', '/dev/null']
    pipe_args = {'stdout' : subprocess.PIPE, 'stderr' : subprocess.PIPE}
    super().__init__(cmd_args, cwd=self.cwd.name, **pipe_args)

  def __exit__(self, *args):
    super().__exit__(*args)
    self.tcl_file.close()
    self.cwd.cleanup()

PACKAGEXO_COMMANDS = r'''
set tmp_ip_dir "{tmpdir}/tmp_ip_dir"
set tmp_project "{tmpdir}/tmp_project"

create_project -force kernel_pack ${{tmp_project}}
add_files -norecurse [glob {hdl_dir}/*.v]
foreach tcl_file [glob -nocomplain {hdl_dir}/*.tcl] {{
  source ${{tcl_file}}
}}
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
ipx::package_project -root_dir ${{tmp_ip_dir}} -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core ${{tmp_ip_dir}}/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory ${{tmp_ip_dir}} ${{tmp_ip_dir}}/component.xml
set_property core_revision 2 [ipx::current_core]
foreach up [ipx::get_user_parameters] {{
  ipx::remove_user_parameter [get_property NAME ${{up}}] [ipx::current_core]
}}
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
{bus_ifaces}
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
set_property xpm_libraries {{XPM_CDC XPM_MEMORY XPM_FIFO}} [ipx::current_core]
set_property supported_families {{ }} [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete

package_xo -force -xo_path "{xo_file}" -kernel_name {top_name} -ip_directory ${{tmp_ip_dir}} -kernel_xml {kernel_xml}{cpp_kernels}
'''

class PackageXo(Vivado):
  """Packages the given files into a Xilinx hardware object.

  Args:
    xo_file: name of the generated xo file.
    top_name: top-level module name.
    kernel_xml: xml description of the kernel.
    hdl_dir: directory of all HDL files.
    m_axi_names: variable names connected to the m_axi bus.
    cpp_kernels: sequence of file names of C++ kernels.
  """
  def __init__(self, xo_file, top_name, kernel_xml, hdl_dir, m_axi_names,
               cpp_kernels=()):
    self.tmpdir = tempfile.TemporaryDirectory(prefix='package-xo-')
    if _logger.isEnabledFor(logging.INFO):
      for _, _, files in os.walk(hdl_dir):
        for filename in files:
          _logger.info('packing: %s', filename)
    kwargs = {
        'top_name' : top_name,
        'kernel_xml' : kernel_xml,
        'hdl_dir' : hdl_dir,
        'xo_file' : xo_file,
        'bus_ifaces' : '\n'.join(map(
            'ipx::associate_bus_interfaces -busif m_axi_{} -clock ap_clk '
            '[ipx::current_core]'.format, m_axi_names)),
        'tmpdir' : self.tmpdir.name,
        'cpp_kernels' : ''.join(map(' -kernel_files {}'.format, cpp_kernels))
    }
    super().__init__(PACKAGEXO_COMMANDS.format(**kwargs))

  def __exit__(self, *args):
    super().__exit__(*args)
    self.tmpdir.cleanup()

HLS_COMMANDS = r'''
cd "{project_dir}"
open_project "{project_name}"
set_top {top_name}
{add_kernels}
open_solution "{solution_name}"
set_part {{{part_num}}}
create_clock -period {clock_period} -name default
config_compile -name_max_length 253
config_interface -m_axi_addr64
config_rtl -disable_start_propagation
csynth_design
exit
'''

class RunHls(VivadoHls):
  """Runs Vivado HLS for the given kernels and generate HDL files

  Args:
    tarfileobj: file object that will contain the reports and HDL files.
    kernel_files: file names of the kernels.
    top_name: top-level module name.
    clock_period: target clock period.
    part_num: target part number.
  """
  def __init__(self, tarfileobj, kernel_files, top_name, clock_period,
               part_num):
    self.project_dir = tempfile.TemporaryDirectory(prefix='hls-')
    self.project_name = 'project'
    self.solution_name = 'solution'
    self.tarfileobj = tarfileobj
    kwargs = {
        'project_dir' : self.project_dir.name,
        'project_name' : self.project_name,
        'solution_name' : self.solution_name,
        'top_name' : top_name,
        'add_kernels' :  '\n'.join(map(
            'add_files "{}" -cflags "-std=c++11"'.format, kernel_files)),
        'part_num' : part_num,
        'clock_period' : clock_period
    }
    super().__init__(HLS_COMMANDS.format(**kwargs))

  def __exit__(self, *args):
    super().__exit__(*args)
    if self.returncode == 0:
      with tarfile.open(mode='w', fileobj=self.tarfileobj) as tar:
        solution_dir = os.path.join(self.project_dir.name, self.project_name,
                                    self.solution_name)
        tar.add(os.path.join(solution_dir, 'syn/report'), arcname='report')
        tar.add(os.path.join(solution_dir, 'syn/verilog'), arcname='hdl')
        tar.add(os.path.join(solution_dir, self.solution_name + '.log'),
                arcname=self.solution_name + '.log')
    self.project_dir.cleanup()

XILINX_XML_NS = {'xd' : 'http://www.xilinx.com/xd'}

def get_device_info(platform_path):
  """Extract device part number and target frequency from SDAccel platform.

  Currently only support 5.x platforms.
  """
  device_name = os.path.basename(platform_path)
  with zipfile.ZipFile(os.path.join(
      platform_path, 'hw', device_name + '.dsa')) as platform:
    with platform.open(device_name + '.hpfm') as metadata:
      platform_info = ET.parse(metadata).find('./xd:component/xd:platformInfo',
                                              XILINX_XML_NS)
      return {
          'clock_period' : platform_info.find(
              "./xd:systemClocks/xd:clock/[@xd:id='0']", XILINX_XML_NS).attrib[
                  '{{{xd}}}period'.format(**XILINX_XML_NS)],
          'part_num' : platform_info.find(
              'xd:deviceInfo', XILINX_XML_NS).attrib[
                  '{{{xd}}}name'.format(**XILINX_XML_NS)]
      }

KERNEL_XML_TEMPLATE = r'''
<?xml version="1.0" encoding="UTF-8"?>
<root versionMajor="1" versionMinor="5">
  <kernel name="{top_name}" language="ip" vlnv="xilinx.com:RTLKernel:{top_name}:1.0" attributes="" preferredWorkGroupSizeMultiple="0" workGroupSize="1" debug="true" compileOptions=" -g" profileType="none">
    <ports>{m_axi_ports}
      <port name="s_axi_control" mode="slave" range="0x1000" dataWidth="32" portType="addressable" base="0x0"/>
    </ports>
    <args>{args}
    </args>
  </kernel>
</root>
'''

PORT_TEMPLATE = r'''
      <port name="m_axi_{name}" mode="master" range="0xFFFFFFFF" dataWidth="{width}" portType="addressable" base="0x0"/>
'''

ARG_TEMPLATE = r'''
      <arg name="{name}" addressQualifier="{addr_qualifier}" id="{arg_id}" port="{port_name}" size="{size:#x}" offset="{offset:#x}" hostOffset="0x0" hostSize="{host_size:#x}" type="{c_type}"/>
'''

def print_kernel_xml(top_name, ports, kernel_xml):
  """Generate kernel.xml file.

  Args:
    top_name: name of the top-level kernel function.
    ports: sequence of (port_name, bundle_name, haoda_type, _) of m_axi ports
    kernel_xml: file object to write to.
  """
  m_axi_ports = ''
  args = ''
  offset = 0x10
  arg_id = 0
  bundle_set = set()
  for port_name, bundle_name, haoda_type, _ in ports:
    size = host_size = 8
    if bundle_name not in bundle_set:
      m_axi_ports += PORT_TEMPLATE.format(
          name=bundle_name,
          width=util.get_width_in_bits(haoda_type)).rstrip('\n')
      bundle_set.add(bundle_name)
    args += ARG_TEMPLATE.format(
        name=port_name, addr_qualifier=1, arg_id=arg_id,
        port_name='m_axi_' + bundle_name, c_type=util.get_c_type(haoda_type),
        size=size, offset=offset, host_size=host_size).rstrip('\n')
    offset += size + 4
    arg_id += 1
  args += ARG_TEMPLATE.format(
      name='coalesced_data_num', addr_qualifier=0, arg_id=arg_id,
      port_name='s_axi_control', c_type='uint64_t', size=size, offset=offset,
      host_size=host_size).rstrip('\n')
  kernel_xml.write(KERNEL_XML_TEMPLATE.format(
      top_name=top_name, m_axi_ports=m_axi_ports, args=args))

BRAM_FIFO_TEMPLATE = r'''
`timescale 1ns/1ps

module {name}_w{width}_d{depth}_A
#(parameter
    MEM_STYLE   = "block",
    DATA_WIDTH  = {width},
    ADDR_WIDTH  = {addr_width},
    DEPTH       = {depth}
)
(
    // system signal
    input  wire                  clk,
    input  wire                  reset,

    // write
    output wire                  if_full_n,
    input  wire                  if_write_ce,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,

    // read
    output wire                  if_empty_n,
    input  wire                  if_read_ce,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout
);
//------------------------Parameter----------------------

//------------------------Local signal-------------------
(* ram_style = MEM_STYLE *)
reg  [DATA_WIDTH-1:0] mem[0:DEPTH-1];
reg  [DATA_WIDTH-1:0] q_buf = 1'b0;
reg  [ADDR_WIDTH-1:0] waddr = 1'b0;
reg  [ADDR_WIDTH-1:0] raddr = 1'b0;
wire [ADDR_WIDTH-1:0] wnext;
wire [ADDR_WIDTH-1:0] rnext;
wire                  push;
wire                  pop;
reg  [ADDR_WIDTH-1:0] usedw = 1'b0;
reg                   full_n = 1'b1;
reg                   empty_n = 1'b0;
reg  [DATA_WIDTH-1:0] q_tmp = 1'b0;
reg                   show_ahead = 1'b0;
reg  [DATA_WIDTH-1:0] dout_buf = 1'b0;
reg                   dout_valid = 1'b0;


//------------------------Instantiation------------------

//------------------------Task and function--------------

//------------------------Body---------------------------
assign if_full_n  = full_n;
assign if_empty_n = dout_valid;
assign if_dout    = dout_buf;
assign push       = full_n & if_write_ce & if_write;
assign pop        = empty_n & if_read_ce & (~dout_valid | if_read);
assign wnext      = !push                ? waddr :
                    (waddr == DEPTH - 1) ? 1'b0  :
                    waddr + 1'b1;
assign rnext      = !pop                 ? raddr :
                    (raddr == DEPTH - 1) ? 1'b0  :
                    raddr + 1'b1;

// waddr
always @(posedge clk) begin
    if (reset == 1'b1)
        waddr <= 1'b0;
    else
        waddr <= wnext;
end

// raddr
always @(posedge clk) begin
    if (reset == 1'b1)
        raddr <= 1'b0;
    else
        raddr <= rnext;
end

// usedw
always @(posedge clk) begin
    if (reset == 1'b1)
        usedw <= 1'b0;
    else if (push & ~pop)
        usedw <= usedw + 1'b1;
    else if (~push & pop)
        usedw <= usedw - 1'b1;
end

// full_n
always @(posedge clk) begin
    if (reset == 1'b1)
        full_n <= 1'b1;
    else if (push & ~pop)
        full_n <= (usedw != DEPTH - 1);
    else if (~push & pop)
        full_n <= 1'b1;
end

// empty_n
always @(posedge clk) begin
    if (reset == 1'b1)
        empty_n <= 1'b0;
    else if (push & ~pop)
        empty_n <= 1'b1;
    else if (~push & pop)
        empty_n <= (usedw != 1'b1);
end

// mem
always @(posedge clk) begin
    if (push)
        mem[waddr] <= if_din;
end

// q_buf
always @(posedge clk) begin
    q_buf <= mem[rnext];
end

// q_tmp
always @(posedge clk) begin
    if (reset == 1'b1)
        q_tmp <= 1'b0;
    else if (push)
        q_tmp <= if_din;
end

// show_ahead
always @(posedge clk) begin
    if (reset == 1'b1)
        show_ahead <= 1'b0;
    else if (push && usedw == pop)
        show_ahead <= 1'b1;
    else
        show_ahead <= 1'b0;
end

// dout_buf
always @(posedge clk) begin
    if (reset == 1'b1)
        dout_buf <= 1'b0;
    else if (pop)
        dout_buf <= show_ahead? q_tmp : q_buf;
end

// dout_valid
always @(posedge clk) begin
    if (reset == 1'b1)
        dout_valid <= 1'b0;
    else if (pop)
        dout_valid <= 1'b1;
    else if (if_read_ce & if_read)
        dout_valid <= 1'b0;
end

endmodule
'''

SRL_FIFO_TEMPLATE = r'''
// ==============================================================
// File generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
//
// ==============================================================


`timescale 1 ns / 1 ps

module {name}_w{width}_d{depth}_A_shiftReg (
    clk,
    data,
    ce,
    a,
    q);

parameter DATA_WIDTH = 32'd{width};
parameter ADDR_WIDTH = 32'd{addr_width};
parameter DEPTH = {depth_width}'d{depth};

input clk;
input [DATA_WIDTH-1:0] data;
input ce;
input [ADDR_WIDTH-1:0] a;
output [DATA_WIDTH-1:0] q;

reg[DATA_WIDTH-1:0] SRL_SIG [0:DEPTH-1];
integer i;

always @ (posedge clk)
    begin
        if (ce)
        begin
            for (i=0;i<DEPTH-1;i=i+1)
                SRL_SIG[i+1] <= SRL_SIG[i];
            SRL_SIG[0] <= data;
        end
    end

assign q = SRL_SIG[a];

endmodule

module {name}_w{width}_d{depth}_A (
    clk,
    reset,
    if_empty_n,
    if_read_ce,
    if_read,
    if_dout,
    if_full_n,
    if_write_ce,
    if_write,
    if_din);

parameter MEM_STYLE   = "shiftreg";
parameter DATA_WIDTH  = 32'd{width};
parameter ADDR_WIDTH  = 32'd{addr_width};
parameter DEPTH       = {depth_width}'d{depth};

input clk;
input reset;
output if_empty_n;
input if_read_ce;
input if_read;
output[DATA_WIDTH - 1:0] if_dout;
output if_full_n;
input if_write_ce;
input if_write;
input[DATA_WIDTH - 1:0] if_din;

wire[ADDR_WIDTH - 1:0] shiftReg_addr ;
wire[DATA_WIDTH - 1:0] shiftReg_data, shiftReg_q;
wire                     shiftReg_ce;
reg[ADDR_WIDTH:0] mOutPtr = ~{{(ADDR_WIDTH+1){{1'b0}}}};
reg internal_empty_n = 0, internal_full_n = 1;

assign if_empty_n = internal_empty_n;
assign if_full_n = internal_full_n;
assign shiftReg_data = if_din;
assign if_dout = shiftReg_q;

always @ (posedge clk) begin
    if (reset == 1'b1)
    begin
        mOutPtr <= ~{{ADDR_WIDTH+1{{1'b0}}}};
        internal_empty_n <= 1'b0;
        internal_full_n <= 1'b1;
    end
    else begin
        if (((if_read & if_read_ce) == 1 & internal_empty_n == 1) &&
            ((if_write & if_write_ce) == 0 | internal_full_n == 0))
        begin
            mOutPtr <= mOutPtr - {depth_width}'d1;
            if (mOutPtr == {depth_width}'d0)
                internal_empty_n <= 1'b0;
            internal_full_n <= 1'b1;
        end
        else if (((if_read & if_read_ce) == 0 | internal_empty_n == 0) &&
            ((if_write & if_write_ce) == 1 & internal_full_n == 1))
        begin
            mOutPtr <= mOutPtr + {depth_width}'d1;
            internal_empty_n <= 1'b1;
            if (mOutPtr == DEPTH - {depth_width}'d2)
                internal_full_n <= 1'b0;
        end
    end
end

assign shiftReg_addr = mOutPtr[ADDR_WIDTH] == 1'b0 ? mOutPtr[ADDR_WIDTH-1:0]:{{ADDR_WIDTH{{1'b0}}}};
assign shiftReg_ce = (if_write & if_write_ce) & internal_full_n;

{name}_w{width}_d{depth}_A_shiftReg
#(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .DEPTH(DEPTH))
U_{name}_w{width}_d{depth}_A_ram (
    .clk(clk),
    .data(shiftReg_data),
    .ce(shiftReg_ce),
    .a(shiftReg_addr),
    .q(shiftReg_q));

endmodule
'''

class VerilogPrinter(util.Printer):
  def module(self, module_name, args):
    self.println('module %s (' % module_name)
    self.do_indent()
    self._out.write(' ' * self._indent * self._tab)
    self._out.write((',\n' + ' ' * self._indent * self._tab).join(args))
    self.un_indent()
    self.println('\n);')

  def endmodule(self, module_name=None):
    if module_name is None:
      self.println('endmodule')
    else:
      self.println('endmodule // %s' % module_name)

  def begin(self):
    self.println('begin')
    self.do_indent()

  def end(self):
    self.un_indent()
    self.println('end')

  def parameter(self, key, value):
    self.println('parameter {} = {};'.format(key, value))

  @contextlib.contextmanager
  def initial(self):
    self.println('initial begin')
    self.do_indent()
    yield
    self.un_indent()
    self.println('end')

  @contextlib.contextmanager
  def always(self, condition):
    self.println('always @ (%s) begin' % condition)
    self.do_indent()
    yield
    self.un_indent()
    self.println('end')

  @contextlib.contextmanager
  def if_(self, condition):
    self.println('if (%s) begin' % condition)
    self.do_indent()
    yield
    self.end()

  def else_(self):
    self.un_indent()
    self.println('end else begin')
    self.do_indent()

  def module_instance(self, module_name, instance_name, args):
    self.println('{module_name} {instance_name}('.format(**locals()))
    self.do_indent()
    if isinstance(args, collections.Mapping):
      self._out.write(',\n'.join(
          ' ' * self._indent * self._tab + '.{}({})'.format(*arg)
          for arg in args.items()))
    else:
      self._out.write(',\n'.join(
          ' ' * self._indent * self._tab + arg for arg in args))
    self.un_indent()
    self.println('\n);')

  def fifo_module(self, width, depth, name='fifo', threshold=1024):
    """Generate FIFO with the given parameters.

    Generate an FIFO module named {name}_w{width}_d{depth}_A. If its capacity
    is larger than threshold, BRAM FIFO will be used. Otherwise, SRL FIFO will
    be used.

    Args:
      printer: VerilogPrinter to print to.
      width: FIFO width
      depth: FIFO depth
      name: Optionally give the fifo a name prefix, default to 'fifo'.
      threshold: Optionally give a threshold to decide whether to use BRAM or
        SRL. Defaults to 1024 bits.
    """
    if width * depth > threshold:
      self.bram_fifo_module(width, depth)
    else:
      self.srl_fifo_module(width, depth)

  def bram_fifo_module(self, width, depth, name='fifo'):
    """Generate BRAM FIFO with the given parameters.

    Generate a BRAM FIFO module named {name}_w{width}_d{depth}_A.

    Args:
      printer: VerilogPrinter to print to.
      width: FIFO width
      depth: FIFO depth
      name: Optionally give the fifo a name prefix, default to 'fifo'.
    """
    self._out.write(BRAM_FIFO_TEMPLATE.format(
        width=width, depth=depth, name=name,
        addr_width=(depth - 1).bit_length()))

  def srl_fifo_module(self, width, depth, name='fifo'):
    """Generate SRL FIFO with the given parameters.

    Generate a SRL FIFO module named {name}_w{width}_d{depth}_A.

    Args:
      printer: VerilogPrinter to print to.
      width: FIFO width
      depth: FIFO depth
      name: Optionally give the fifo a name prefix, default to 'fifo'.
    """
    addr_width = (depth - 1).bit_length()
    self._out.write(SRL_FIFO_TEMPLATE.format(
        width=width, depth=depth, name=name, addr_width=addr_width,
        depth_width=addr_width + 1))
