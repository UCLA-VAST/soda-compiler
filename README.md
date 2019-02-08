# SODA Compiler
Stencil with Optimized Dataflow Architecture Compiler

## Publication

+ Yuze Chi, Jason Cong, Peng Wei, Peipei Zhou. [SODA: Stencil with Optimized Dataflow Architecture](https://doi.org/10.1145/3240765.3240850). In *ICCAD*, 2018. (Best Paper Candidate) [[PDF]](https://about.blaok.me/pub/iccad18.pdf) [[Slides]](https://about.blaok.me/pub/iccad18.slides.pdf)

## SODA DSL Example

    # comments start with hashtag(#)
    
    kernel: blur      # the kernel name, will be used as the kernel name in HLS
    burst width: 512  # DRAM burst I/O width in bits, for Xilinx platform by default it's 512
    unroll factor: 16 # how many pixels are generated per cycle
    
    # specify the dram bank, type, name, and dimension of the input tile
    # the last dimension is not needed and a placeholder '*' must be given
    # dram bank is optional
    # multiple inputs can be specified but 1 and only 1 must specify the dimensions
    input dram 1 uint16: input(2000, *)
    
    # specify an intermediate stage of computation, may appear 0 or more times
    local uint16: blur_x(0, 0) = (input(0, 0) + input(0, 1) + input(0, 2)) / 3
    
    # specify the output
    # dram bank is optional
    output dram 1 uint16: blur_y(0, 0) = (blur_x(0, 0) + blur_x(1, 0) + blur_x(2, 0)) / 3
    
    # how many times the whole computation is repeated (only works if input matches output)
    iterate: 1

## Getting Started

### Prerequisites

+ Python 3.5+ and corresponding `pip`
+ SDAccel 2018.3 (earlier versions might work but won't be supported)

<details><summary>How to install Python 3.5+ on Ubuntu 16.04+ and CentOS 7?</summary>
  
#### Ubuntu 16.04+
```bash
sudo apt install python3 python3-pip
```

#### CentOS 7
```bash
sudo yum install python36 python36-pip
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 100
```

</details>

### Clone the Repo
    git clone https://github.com/UCLA-VAST/soda-compiler.git
    cd soda-compiler
    python3 -m pip install --user -r requirements.txt

### Parameter Setup
    app=blur
    platform=xilinx_u200_xdma_201830_1
    # The following can be set via sourcing /path/to/xilinx/sdx/settings64.sh
    XILINX_SDX=/path/to/xilinx/sdx
    XILINX_VIVADO=/path/to/xilinx/vivado

### Generate HLS Kernel Code
    src/sodac tests/src/${app}.soda --xocl-kernel ${app}_kernel.cpp

### Generate OpenCL Host Code
    src/sodac tests/src/${app}.soda --xocl-header ${app}.h
    src/sodac tests/src/${app}.soda --xocl-host ${app}.cpp

### Create Testbench
    cat >${app}_run.cpp <<EOF
    #include <cstdio>
    #include <cstdlib>
    
    #include "${app}.h"                                                                            
    
    int ${app}_test(const char* xclbin, const int dims[4]);
    int main(int argc, char **argv) {
      if (argc != 4) {
        fprintf(stderr, "Usage: \n    %s <xclbin> <input width> <input height>\n", argv[0]);
        return 1;
      }
      int dims[4] = {atoi(argv[2]), atoi(argv[3]), 0, 0};
      return ${app}_test(argv[1], dims);
    }
    EOF

### Compile OpenCL Host Executable
    # Please set TILE_SIZE_DIM_0 and UNROLL_FACTOR macros to match the kernel.
    g++ -std=c++11 -I${XILINX_SDX}/runtime/include -I${XILINX_VIVADO}/include ${app}.cpp ${app}_run.cpp -o ${app} \
        -lxilinxopencl -DTILE_SIZE_DIM_0=2000 -DUNROLL_FACTOR=2 -fopenmp -Wno-deprecated-declarations -Wall

### Create Emulation Config
    emconfigutil -f ${platform}

### Software Emulation

#### Compile for Software Emulation
    xocc -t sw_emu -f ${platform} --kernel ${app}_kernel --xp prop:kernel.${app}_kernel.kernel_flags="-std=c++0x" \
        -c ${app}_kernel.cpp -o ${app}.sw_emu.xo

#### Link for Software Emulation
    xocc -t sw_emu -f ${platform} -l ${app}.sw_emu.xo -o ${app}.sw_emu.xclbin

#### Run Software Emulation
    XCL_EMULATION_MODE=sw_emu ./${app} ${app}.sw_emu.xclbin 2000 100

### High-Level Synthesis
    xocc -t hw -f ${platform} --kernel ${app}_kernel --xp prop:kernel.${app}_kernel.kernel_flags="-std=c++0x" \
        -c ${app}_kernel.cpp -o ${app}.hw.xo

### Hardware Emulation

#### Link for Hardware Emulation
    xocc -t hw_emu -f ${platform} -l ${app}.hw.xo -o ${app}.hw_emu.xclbin

#### Run Hardware Emulation
    # By default, kernel ports are connected via DRAM bank 1 on the xilinx_u200_xdma_201830_1 platform.
    DRAM_IN=1 DRAM_OUT=1 XCL_EMULATION_MODE=hw_emu ./${app} ${app}.hw_emu.xclbin 2000 10

### Hardware Deployment

#### Logic Synthesis, Place, and Route
    xocc -t hw -f ${platform} -l ${app}.hw.xo -o ${app}.hw.xclbin

#### Run Bitstream on FPGA
    # By default, kernel ports are connected via DRAM bank 1 on the xilinx_u200_xdma_201830_1 platform.
    DRAM_IN=1 DRAM_OUT=1 ./${app} ${app}.hw.xclbin 2000 1000

## Code Snippet Example

### Source Code

    kernel: jacobi2d
    burst width: 512
    unroll factor: 2
    input float: t1(2000, *)
    output float: t0(0, 0) = (t1(0, 1) + t1(1, 0) + t1(0, 0) + t1(0, -1) + t1(-1, 0)) * 0.2f
    iterate: 1

### HLS Kernel Code
Each function in the below code snippets is synthesized into an RTL module.
Their arguments are all `hls::stream` FIFOs; Without unrolling, a simple line-buffer pipeline is generated, producing 1 pixel per cycle.
With unrolling, a SODA microarchitecture pipeline is generated, procuding 2 pixeles per cycle.

#### Without Unrolling (`--unroll-factor=1`)

    #pragma HLS dataflow
    Module1Func(
      /*output*/ &from_t1_offset_0_to_t1_offset_1999,
      /*output*/ &from_t1_offset_0_to_t0_pe_0,
      /* input*/ &from_super_source_to_t1_offset_0);
    Module2Func(
      /*output*/ &from_t1_offset_1999_to_t1_offset_2000,
      /*output*/ &from_t1_offset_1999_to_t0_pe_0,
      /* input*/ &from_t1_offset_0_to_t1_offset_1999);
    Module3Func(
      /*output*/ &from_t1_offset_2000_to_t1_offset_2001,
      /*output*/ &from_t1_offset_2000_to_t0_pe_0,
      /* input*/ &from_t1_offset_1999_to_t1_offset_2000);
    Module3Func(
      /*output*/ &from_t1_offset_2001_to_t1_offset_4000,
      /*output*/ &from_t1_offset_2001_to_t0_pe_0,
      /* input*/ &from_t1_offset_2000_to_t1_offset_2001);
    Module4Func(
      /*output*/ &from_t1_offset_4000_to_t0_pe_0,
      /* input*/ &from_t1_offset_2001_to_t1_offset_4000);
    Module5Func(
      /*output*/ &from_t0_pe_0_to_super_sink,
      /* input*/ &from_t1_offset_0_to_t0_pe_0,
      /* input*/ &from_t1_offset_1999_to_t0_pe_0,
      /* input*/ &from_t1_offset_2000_to_t0_pe_0,
      /* input*/ &from_t1_offset_4000_to_t0_pe_0,
      /* input*/ &from_t1_offset_2001_to_t0_pe_0);

In the above code snippet, `Module1Func` to `Module4Func` are forwarding modules; they constitute the data-reuse line buffer.
The line buffer size is approximately two lines of pixels, i.e. 4000 pixels.
`Module5Func` is a computing module; it implements the computation kernel.
The whole design is fully pipelined; however, with only 1 computing module, it can only produce 1 pixel per cycle.

#### Unroll 2 Times (`--unroll-factor=2`)

    #pragma HLS dataflow
    Module1Func(
      /*output*/ &from_t1_offset_1_to_t1_offset_1999,
      /*output*/ &from_t1_offset_1_to_t0_pe_0,
      /* input*/ &from_super_source_to_t1_offset_1);
    Module1Func(
      /*output*/ &from_t1_offset_0_to_t1_offset_2000,
      /*output*/ &from_t1_offset_0_to_t0_pe_1,
      /* input*/ &from_super_source_to_t1_offset_0);
    Module2Func(
      /*output*/ &from_t1_offset_1999_to_t1_offset_2001,
      /*output*/ &from_t1_offset_1999_to_t0_pe_1,
      /* input*/ &from_t1_offset_1_to_t1_offset_1999);
    Module3Func(
      /*output*/ &from_t1_offset_2000_to_t1_offset_2002,
      /*output*/ &from_t1_offset_2000_to_t0_pe_1,
      /*output*/ &from_t1_offset_2000_to_t0_pe_0,
      /* input*/ &from_t1_offset_0_to_t1_offset_2000);
    Module4Func(
      /*output*/ &from_t1_offset_2001_to_t1_offset_4001,
      /*output*/ &from_t1_offset_2001_to_t0_pe_1,
      /*output*/ &from_t1_offset_2001_to_t0_pe_0,
      /* input*/ &from_t1_offset_1999_to_t1_offset_2001);
    Module5Func(
      /*output*/ &from_t1_offset_2002_to_t1_offset_4000,
      /*output*/ &from_t1_offset_2002_to_t0_pe_0,
      /* input*/ &from_t1_offset_2000_to_t1_offset_2002);
    Module6Func(
      /*output*/ &from_t1_offset_4001_to_t0_pe_0,
      /* input*/ &from_t1_offset_2001_to_t1_offset_4001);
    Module7Func(
      /*output*/ &from_t0_pe_0_to_super_sink,
      /* input*/ &from_t1_offset_1_to_t0_pe_0,
      /* input*/ &from_t1_offset_2000_to_t0_pe_0,
      /* input*/ &from_t1_offset_2001_to_t0_pe_0,
      /* input*/ &from_t1_offset_4001_to_t0_pe_0,
      /* input*/ &from_t1_offset_2002_to_t0_pe_0);
    Module8Func(
      /*output*/ &from_t1_offset_4000_to_t0_pe_1,
      /* input*/ &from_t1_offset_2002_to_t1_offset_4000);
    Module7Func(
      /*output*/ &from_t0_pe_1_to_super_sink,
      /* input*/ &from_t1_offset_0_to_t0_pe_1,
      /* input*/ &from_t1_offset_1999_to_t0_pe_1,
      /* input*/ &from_t1_offset_2000_to_t0_pe_1,
      /* input*/ &from_t1_offset_4000_to_t0_pe_1,
      /* input*/ &from_t1_offset_2001_to_t0_pe_1);

In the above code snippet, `Module1Func` to `Module6Func` and `Module8Func` are forwarding modules; they constitute the reuse buffers of the SODA microarchitecture.
Although unrolled, the reuse buffer size is still approximately two lines of pixels, i.e. 4000 pixels.
`Module7Func` is a computing module; it is instanciated twice.
The whole design is fully pipelined and can produce 2 pixel per cycle.
In general, the unroll factor can be set to any number that satisfies the throughput requirement.

## Design Considerations

+ `kernel`, `burst width`, `unroll factor`, `input`, `output`, and `iterate` keywords are mandatory
+ For non-iterative stencil, `unroll factor` shall be determined by the DRAM bandwidth, i.e. saturate the external bandwidth, since the resource is usually not the bottleneck
+ For iterative stencil, prefer to use more PEs in a single iteration rather than implement more iterations
+ Note that `2.0` will be a `double` number. To generate `float`, use `2.0f`. This may help reduce DSP usage
+ SODA is tiling-based and the size of the tile is specified in the `input` keyword. The last dimension is a placeholder because it is not needed in the reuse buffer generation

## Projects Using SODA

+ Yi-Hsiang Lai, Yuze Chi, Yuwei Hu, Jie Wang, Cody Hao Yu, Yuan Zhou, Jason Cong, Zhiru Zhang. [HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing](https://doi.org/10.1145/3289602.3293910). In *FPGA*, 2019. (Best Paper Candidate) [[PDF]](https://about.blaok.me/pub/fpga19-heterocl.pdf) [[Slides]](https://about.blaok.me/pub/fpga19-heterocl.slides.pdf)
+ Yuze Chi, Young-kyu Choi, Jason Cong, Jie Wang. [Rapid Cycle-Accurate Simulator for High-Level Synthesis](https://doi.org/10.1145/3289602.3293918). In *FPGA*, 2019. [[PDF]](https://about.blaok.me/pub/fpga19-flash.pdf) [[Slides]](https://about.blaok.me/pub/fpga19-flash.slides.pdf)
