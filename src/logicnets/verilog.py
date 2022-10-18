#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np

def generate_register_verilog(module_name="myreg", param_name="DataWidth", input_name="data_in", output_name="data_out"):
    register_template = """\
module {module_name} #(parameter {param_name}=16) (
    input [{param_name}-1:0] {input_name},
    input wire clk,
    input wire rst,
    output reg [{param_name}-1:0] {output_name}
    );
    always@(posedge clk) begin
    if(!rst)
        {output_name}<={input_name};
    else
        {output_name}<=0;
    end
endmodule\n
"""
    return register_template.format(    module_name=module_name,
                                        param_name=param_name,
                                        input_name=input_name,
                                        output_name=output_name)

def generate_logicnets_verilog(module_name: str, input_name: str, input_bits: int, output_name: str, output_bits: int, module_contents: str):
    logicnets_template = """\
module {module_name} (input [{input_bits_1:d}:0] {input_name}, input clk, input rst, output[{output_bits_1:d}:0] {output_name});
{module_contents}
endmodule\n"""
    return logicnets_template.format( module_name=module_name,
                                input_name=input_name,
                                input_bits_1=input_bits-1,
                                output_name=output_name,
                                output_bits_1=output_bits-1,
                                module_contents=module_contents)

def layer_connection_verilog(layer_string: str, input_string: str, input_bits: int, output_string: str, output_bits: int, output_wire=True, register=False):
    if register:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
myreg #(.DataWidth({input_bits})) {layer_string}_reg (.data_in({input_string}), .clk(clk), .rst(rst), .data_out({input_string}w));\n"""
    else:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
assign {input_string}w = {input_string};\n"""
    layer_connection_template += "wire [{output_bits_1:d}:0] {output_string};\n" if output_wire else ""
    layer_connection_template += "{layer_string} {layer_string}_inst (.M0({input_string}w), .M1({output_string}));\n"
    return layer_connection_template.format(    layer_string=layer_string,
                                                input_string=input_string,
                                                input_bits=input_bits,
                                                input_bits_1=input_bits-1,
                                                output_string=output_string,
                                                output_bits_1=output_bits-1)

def generate_lut_verilog(module_name, input_fanin_bits, output_bits, lut_string):
    lut_neuron_template = """\
module {module_name} ( input [{input_fanin_bits_1:d}:0] M0, output [{output_bits_1:d}:0] M1 );

	(*rom_style = "distributed" *) reg [{output_bits_1:d}:0] M1r;
	assign M1 = M1r;
	always @ (M0) begin
		case (M0)
{lut_string}
		endcase
	end
endmodule\n"""
    return lut_neuron_template.format(  module_name=module_name,
                                        input_fanin_bits_1=input_fanin_bits-1,
                                        output_bits_1=output_bits-1,
                                        lut_string=lut_string)

def generate_neuron_connection_verilog(input_indices, input_bitwidth):
    connection_string = ""
    for i in range(len(input_indices)):
        index = input_indices[i]
        offset = index*input_bitwidth
        for b in reversed(range(input_bitwidth)):
            connection_string += f"M0[{offset+b}]"
            if not (i == len(input_indices)-1 and b == 0):
                connection_string += ", "
    return connection_string

def fix_abc_module_name(input_verilog_file, output_verilog_file, old_module_name, new_module_name, add_timescale: bool = False):
    with open(input_verilog_file, 'r') as f:
        lines = f.readlines()
    with open(output_verilog_file, 'w') as f:
        if add_timescale:
            f.write("`timescale 1 ps / 1 ps\n")
        for l in lines:
            if l.__contains__(f"module {old_module_name}"):
                l = f"module {new_module_name}  (\n"
            f.write(l)

def generate_abc_verilog_wrapper(module_name: str, input_name: str, input_bits: int, output_name: str, output_bits: int, submodule_name: str, num_registers: int, add_timescale: bool = True):
    abc_wrapper_template = """\
{timescale}
module {module_name} (input [{input_bits_1:d}:0] {input_name}, input clk, input rst, output[{output_bits_1:d}:0] {output_name});
{module_contents}
endmodule\n"""
    input_digits = int(np.ceil(np.log10(input_bits)))
    output_digits = int(np.ceil(np.log10(output_bits)))
    module_contents = []
    module_contents.append(f"{submodule_name} {submodule_name}_inst (")
    # Connect inputs
    if num_registers > 0:
        module_contents.append(f"    .clock(clk),")
    for i in range(input_bits):
        module_contents.append(f"    .pi{i:0{input_digits}d}({input_name}[{i}]),")
    for i in range(output_bits):
        if i < output_bits-1:
            module_contents.append(f"    .po{i:0{output_digits}d}({output_name}[{i}]),")
        else:
            module_contents.append(f"    .po{i:0{output_digits}d}({output_name}[{i}])")
    module_contents.append(f"    );\n")
    module_contents = "\n".join(module_contents)
    return abc_wrapper_template.format( module_name=module_name,
                                input_name=input_name,
                                input_bits_1=input_bits-1,
                                output_name=output_name,
                                output_bits_1=output_bits-1,
                                module_contents=module_contents,
                                timescale="`timescale 1 ps / 1 ps" if add_timescale else "")

