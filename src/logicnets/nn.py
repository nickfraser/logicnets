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

from functools import partial, reduce

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .init import random_restrict_fanin
from .util import fetch_mask_indices, generate_permutation_matrix
from .verilog import    generate_lut_verilog, \
                        generate_neuron_connection_verilog, \
                        layer_connection_verilog, \
                        generate_logicnets_verilog, \
                        generate_register_verilog
from .bench import      generate_lut_bench, \
                        generate_lut_input_string, \
                        sort_to_bench

# TODO: Create a container module which performs this function.
# Generate all truth tables for NEQs for a given nn.Module()
def generate_truth_tables(model: nn.Module, verbose: bool = False) -> None:
    training = model.training
    model.eval()
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            if verbose:
                print(f"Calculating truth tables for {name}")
            module.calculate_truth_tables()
            if verbose:
                print(f"Truth tables generated for {len(module.neuron_truth_tables)} neurons")
    model.training = training

# TODO: Create a container module which performs this function.
def lut_inference(model: nn.Module, track_used_luts: bool = False) -> None:
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            module.lut_inference(track_used_luts=track_used_luts)

# TODO: Create a container module which performs this function.
def save_luts(model: nn.Module, path: str) -> None:
    lut_dict = {}
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            luts = module.neuron_truth_tables
            indices = list(map(lambda x: x[0], luts))
            tt_inputs = list(map(lambda x: x[1], luts))
            tt_input_bin_str = list(map(lambda x: list(map(lambda y: list(map(lambda z: module.input_quant.get_bin_str(z), y)), x)), tt_inputs))
            tt_float_outputs = list(map(lambda x: x[2], luts))
            tt_bin_outputs = list(map(lambda x: x[3], luts))
            tt_outputs_bin_str = list(map(lambda x: list(map(lambda y: module.output_quant.get_bin_str(y), x)), tt_bin_outputs))
            histogram = module.used_luts_histogram
            lut_dict[name] = {
                'indices': indices,
                'input_state_space': tt_inputs,
                'input_state_space_bin_str': tt_input_bin_str,
                'output_state_space_float': tt_float_outputs,
                'output_state_space_bin': tt_bin_outputs,
                'output_state_space_bin_str': tt_outputs_bin_str,
                'histogram': histogram,
            }
    torch.save(lut_dict, path)

# TODO: Create a container module which performs this function.
def load_histograms(model: nn.Module, lut_dict: dict) -> None:
    for name, module in model.named_modules():
        if name in lut_dict.keys():
            module.used_luts_histogram = lut_dict[name]['histogram']

# TODO: Create a container module which performs this function.
def neq_inference(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            module.neq_inference()

# TODO: Should this go in with the other verilog functions?
# TODO: Support non-linear topologies
def module_list_to_verilog_module(module_list: nn.ModuleList, module_name: str, output_directory: str, add_registers: bool = True, generate_bench: bool = True, freq_thresh = None):
    input_bitwidth = None
    output_bitwidth = None
    module_contents = ""
    for i in range(len(module_list)):
        m = module_list[i]
        if type(m) == SparseLinearNeq:
            module_prefix = f"layer{i}"
            module_input_bits, module_output_bits = m.gen_layer_verilog(module_prefix, output_directory, freq_thresh=freq_thresh, generate_bench=generate_bench)
            if i == 0:
                input_bitwidth = module_input_bits
            if i == len(module_list)-1:
                output_bitwidth = module_output_bits
            module_contents += layer_connection_verilog( module_prefix,
                                                        input_string=f"M{i}",
                                                        input_bits=module_input_bits,
                                                        output_string=f"M{i+1}",
                                                        output_bits=module_output_bits,
                                                        output_wire=i!=len(module_list)-1,
                                                        register=add_registers)
        else:
            raise Exception(f"Expect type(module) == SparseLinearNeq, {type(module)} found")
    module_list_verilog = generate_logicnets_verilog(   module_name=module_name,
                                                        input_name="M0",
                                                        input_bits=input_bitwidth,
                                                        output_name=f"M{len(module_list)}",
                                                        output_bits=output_bitwidth,
                                                        module_contents=module_contents)
    reg_verilog = generate_register_verilog()
    with open(f"{output_directory}/myreg.v", "w") as f:
        f.write(reg_verilog)
    with open(f"{output_directory}/{module_name}.v", "w") as f:
        f.write(module_list_verilog)

class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: nn.Module, bias: bool = True) -> None:
        super(SparseLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.mask = mask

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight*self.mask(), self.bias)

# TODO: Perhaps make this two classes, separating the LUT and NEQ code.
class SparseLinearNeq(nn.Module):
    def __init__(self, in_features: int, out_features: int, input_quant, output_quant, sparse_linear_kws={}, apply_input_quant=True, apply_output_quant=True) -> None:
        super(SparseLinearNeq, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.fc = SparseLinear(in_features, out_features, **sparse_linear_kws)
        self.output_quant = output_quant
        self.is_lut_inference = False
        self.neuron_truth_tables = None
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant
        self.track_used_luts = False
        self.used_luts_histogram = None

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    # TODO: Update this code to support custom bitwidths per input/output
    def gen_layer_verilog(self, module_prefix, directory, freq_thresh = None, generate_bench: bool = True):
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.in_features*input_bitwidth
        total_output_bits = self.out_features*output_bitwidth
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output [{total_output_bits-1}:0] M1);\n\n"
        output_offset = 0
        for index in range(self.out_features):
            module_name = f"{module_prefix}_N{index}"
            indices, _, _, _ = self.neuron_truth_tables[index]
            neuron_verilog = self.gen_neuron_verilog(index, module_name, freq_thresh=freq_thresh) # Generate the contents of the neuron verilog
            with open(f"{directory}/{module_name}.v", "w") as f:
                f.write(neuron_verilog)
            if generate_bench:
                neuron_bench = self.gen_neuron_bench(index, module_name) # Generate the contents of the neuron verilog
                with open(f"{directory}/{module_name}.bench", "w") as f:
                    f.write(neuron_bench)
            connection_string = generate_neuron_connection_verilog(indices, input_bitwidth) # Generate the string which connects the synapses to this neuron
            wire_name = f"{module_name}_wire"
            connection_line = f"wire [{len(indices)*input_bitwidth-1}:0] {wire_name} = {{{connection_string}}};\n"
            inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}]));\n\n"
            layer_contents += connection_line + inst_line
            output_offset += output_bitwidth
        layer_contents += "endmodule"
        with open(f"{directory}/{module_prefix}.v", "w") as f:
            f.write(layer_contents)
        return total_input_bits, total_output_bits

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    def gen_neuron_verilog(self, index, module_name, freq_thresh=None):
        indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices)*input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        for i in range(num_entries):
            entry_str = ""
            for idx in range(len(indices)):
                val = input_perm_matrix[i,idx]
                entry_str += self.input_quant.get_bin_str(val)
            res_str = self.output_quant.get_bin_str(bin_output_states[i])
            if (freq_thresh is None) or (self.used_luts_histogram[index][i] >= freq_thresh):
                lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: M1r = {int(output_bitwidth)}'b{res_str};\n"
        # Add a default "don't care" statement
        default_string = int(output_bitwidth) * 'x'
        lut_string += f"\t\t\tdefault: M1r = {int(output_bitwidth)}'b{default_string};\n"
        return generate_lut_verilog(module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string)

    # TODO: Move the string templates to bench.py
    # TODO: Move this to another class
    def gen_neuron_bench(self, index, module_name):
        indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices)*input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        # Sort the input_perm_matrix to match the bench format
        input_state_space_bin_str = list(map(lambda y: list(map(lambda z: self.input_quant.get_bin_str(z), y)), input_perm_matrix))
        sorted_bin_output_states = sort_to_bench(input_state_space_bin_str, bin_output_states)
        # Generate the LUT for each output
        for i in range(int(output_bitwidth)):
            lut_string += f"M1[{i}]       = LUT 0x"
            output_bin_str = reduce(lambda b,c: b+c, map(lambda a: self.output_quant.get_bin_str(a)[int(output_bitwidth)-1-i], sorted_bin_output_states))
            lut_hex_string = f"{int(output_bin_str,2):0{int(num_entries/4)}x} "
            lut_string += lut_hex_string
            lut_string += generate_lut_input_string(int(cat_input_bitwidth))
        return generate_lut_bench(int(cat_input_bitwidth), int(output_bitwidth), lut_string)

    def lut_inference(self, track_used_luts=False):
        self.is_lut_inference = True
        self.track_used_luts = track_used_luts
        self.input_quant.bin_output()
        self.output_quant.bin_output()

    def neq_inference(self):
        self.is_lut_inference = False
        self.input_quant.float_output()
        self.output_quant.float_output()

    # TODO: This function might be a useful utility outside of this class..
    def table_lookup(self, connected_input: Tensor, input_perm_matrix: Tensor, bin_output_states: Tensor, neuron_lut_histogram=None) -> Tensor:
        fan_in_size = connected_input.shape[1]
        ci_bcast = connected_input.unsqueeze(2) # Reshape to B x Fan-in x 1
        pm_bcast = input_perm_matrix.t().unsqueeze(0) # Reshape to 1 x Fan-in x InputStates
        eq = (ci_bcast == pm_bcast).sum(dim=1) == fan_in_size # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1) # Count the number of perfect matches per input vector
        if not (matches == torch.ones_like(matches,dtype=matches.dtype)).all():
            raise Exception(f"One or more vectors in the input is not in the possible input state space")
        indices = torch.argmax(eq.type(torch.int64),dim=1)
        if self.track_used_luts:
            # TODO: vectorize this loop
            for i in indices:
                neuron_lut_histogram[i] += 1
        return bin_output_states[indices]

    def lut_forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x) # Use this to fetch the bin output of the input, if the input isn't already in binary format
        # TODO: Put this in a child class(?)
        # TODO: Add support for non-uniform fan-in
        if self.track_used_luts:
            if self.used_luts_histogram is None:
                self.used_luts_histogram = self.out_features * [None]
                for i in range(self.out_features):
                    self.used_luts_histogram[i] = torch.zeros(size=(len(self.neuron_truth_tables[i][2]),), dtype=torch.int64)
        y = torch.zeros((x.shape[0],self.out_features))
        # Perform table lookup for each neuron output
        for i in range(self.out_features):
            indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[i]
            neuron_lut_histogram = self.used_luts_histogram[i] if self.track_used_luts else None
            connected_input = x[:,indices]
            y[:,i] = self.table_lookup(connected_input, input_perm_matrix, bin_output_states, neuron_lut_histogram=neuron_lut_histogram)
        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.is_lut_inference:
            x = self.lut_forward(x)
        else:
            if self.apply_input_quant:
                x = self.input_quant(x)
            x = self.fc(x)
            if self.apply_output_quant:
                x = self.output_quant(x)
        return x

    # Consider using masked_select instead of fetching the indices
    def calculate_truth_tables(self):
        with torch.no_grad():
            mask = self.fc.mask()
            # Precalculate all of the input value permutations
            input_state_space = list() # TODO: is a list the right data-structure here?
            bin_state_space = list()
            for m in range(self.in_features):
                neuron_state_space = self.input_quant.get_state_space() # TODO: this call should include the index of the element of interest
                bin_space = self.input_quant.get_bin_state_space() # TODO: this call should include the index of the element of interest
                input_state_space.append(neuron_state_space)
                bin_state_space.append(bin_space)

            neuron_truth_tables = list()
            for n in range(self.out_features):
                # Determine the fan-in as number of synapse connections
                input_mask = mask[n,:]
                fan_in = torch.sum(input_mask)
                indices = fetch_mask_indices(input_mask)
                # Retrieve the possible state space of the current neuron
                connected_state_space = [input_state_space[i] for i in indices]
                bin_connected_state_space = [bin_state_space[i] for i in indices]
                # Generate a matrix containing all possible input states
                input_permutation_matrix = generate_permutation_matrix(connected_state_space)
                bin_input_permutation_matrix = generate_permutation_matrix(bin_connected_state_space)
                num_permutations = input_permutation_matrix.shape[0]
                padded_perm_matrix = torch.zeros((num_permutations, self.in_features))
                padded_perm_matrix[:,indices] = input_permutation_matrix

                # TODO: Update this block to just run inference on the fc layer, once BN has been moved to output_quant
                apply_input_quant, apply_output_quant = self.apply_input_quant, self.apply_output_quant
                self.apply_input_quant, self.apply_output_quant = False, False
                is_bin_output = self.output_quant.is_bin_output
                self.output_quant.float_output()
                output_states = self.output_quant(self.forward(padded_perm_matrix))[:,n] # Calculate float for the current input
                self.output_quant.bin_output()
                bin_output_states = self.output_quant(self.forward(padded_perm_matrix))[:,n] # Calculate bin for the current input
                self.output_quant.is_bin_output = is_bin_output
                self.apply_input_quant, self.apply_output_quant = apply_input_quant, apply_output_quant

                # Append the connectivity, input permutations and output permutations to the neuron truth tables 
                neuron_truth_tables.append((indices, bin_input_permutation_matrix, output_states, bin_output_states)) # Change this to be the binary output states
        self.neuron_truth_tables = neuron_truth_tables

class DenseMask2D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DenseMask2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 1.0)

    def forward(self):
        return self.mask

class RandomFixedSparsityMask2D(nn.Module):
    def __init__(self, in_features: int, out_features: int, fan_in: int) -> None:
        super(RandomFixedSparsityMask2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in = fan_in
        self.mask = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 0.0)
        for i in range(self.out_features):
            x = torch.randperm(self.in_features)[:self.fan_in]
            self.mask[i][x] = 1

    def forward(self):
        return self.mask

class ScalarScaleBias(nn.Module):
    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0) -> None:
        super(ScalarScaleBias, self).__init__()
        if scale:
            self.weight = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.weight_init = scale_init
        self.bias_init = bias_init
        self.reset_parameters()

    # Change the default initialisation for BatchNorm
    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        if self.weight is not None:
            x = x*self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

class ScalarBiasScale(ScalarScaleBias):
    def forward(self, x):
        if self.bias is not None:
            x = x + self.bias
        if self.weight is not None:
            x = x*self.weight
        return x

