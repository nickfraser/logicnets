#  Copyright (C) 2021 Xilinx, Inc
#  Copyright (C) 2021 Alan Mishchenko
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

def generate_prepare_script_string(num_layers, path):
    prepare_script_template = """\
# This script prepares experiments in ABC by deriving intermediate simulation patterns

# Assuming that verilog/BENCH for each layer of the network are in files "ver/layer{{0,1,2,..}}.v"
# and input/output patterns are the network are in files {{train,test}}_{{input,output}}.txt


# ====================================================================================
# Read the layers from Verilog/BENCH files
{read_layers_string}

# ====================================================================================
# Convert input patterns into the internal binary representation
&lnetread {path}/train_input.txt {path}/train.sim
&lnetread {path}/test_input.txt  {path}/test.sim


# ====================================================================================
# Generate training simulation info for the inputs of each layer
{simulate_layers_string}

# ====================================================================================
# Combine all layers into one monolithic AIG for the whole network (layers.aig)
{gen_monolithic_aig_string}
"""
    read_layer_template = "&lnetread {path}/ver/layer{i}.v; &ps; &w {path}/layer{i}.aig\n"
    simulate_layer_template = "&r {path}/layer{i}.aig; &lnetsim {path}/train{it}.sim {path}/train{ip1}.sim\n"
    gen_monolithic_aig_template = "putontop {layers_aig_string}; st; ps; write {path}/layers.aig\n"
    read_layers_string = ""
    simulate_layers_string = ""
    layers_aig_string = ""
    for i in range(num_layers):
        read_layers_string += read_layer_template.format(i=i, path=path)
        simulate_layers_string += simulate_layer_template.format(i=i, it="" if i == 0 else i, ip1=i+1, path=path)
        layers_aig_string += "{path}/layer{i}.aig ".format(i=i, path=path)
    gen_monolithic_aig_string = gen_monolithic_aig_template.format(layers_aig_string=layers_aig_string.strip(), path=path)
    return prepare_script_template.format(  path=path,
                                            read_layers_string=read_layers_string,
                                            simulate_layers_string=simulate_layers_string,
                                            gen_monolithic_aig_string=gen_monolithic_aig_string)


def generate_opt_script_string(model, path, num_registers, rarity=0):
    opt_script_template = """\
# Generating script with rarity = {rarity}.

# ---- rarity = {rarity} -------------------------------------------------------------------------------------------------------
{optimise_with_rarity_string}

{gen_monolithic_aig_string}

{technology_map_layers_string}

{gen_monolithic_blif_string}

read {path}/blif/layers_opt.blif; ps; pipe -L {num_registers}; ps; retime -M 4; ps; sweep; ps; write_verilog -fm {path}/ver/layers_opt_p{num_registers}.v

&r {path}/aig/layers_opt.aig; &lnetsim {path}/train.sim {path}/train.simo
&r {path}/aig/layers_opt.aig; &lneteval -O 2 {path}/train.simo {path}/train_output.txt

&r {path}/aig/layers_opt.aig; &lnetsim {path}/test.sim {path}/test.simo
&r {path}/aig/layers_opt.aig; &lneteval -O 2 {path}/test.simo {path}/test_output.txt

"""
    optimise_with_rarity_template = "&r {path}/layer{i}.aig; &ps; &lnetopt -I {fanin_bits} -O {fanout_bits} -R {rarity} {path}/train{it}.sim;  &w {path}/aig/layer{i}_opt.aig; &ps; time\n"
    technology_map_layer_template = "&r {path}/aig/layer{i}_opt.aig; &lnetmap -I {fanin_bits} -O {fanout_bits}; write {path}/blif/layer{i}_opt.blif; write_verilog -fm {path}/ver/layer{i}_opt.v\n"
    gen_monolithic_aig_template = "putontop {aig_layers_string}; st; ps; write {path}/aig/layers_opt.aig\n"
    gen_monolithic_blif_template = "putontop {blif_layers_string}; sw; ps; write {path}/blif/layers_opt.blif\n"
    num_layers = 5 # TODO: fetch number of layers from the model
    optimise_with_rarity_string = ""
    technology_map_layers_string = ""
    aig_layers_string = ""
    blif_layers_string = ""
    for i in range(num_layers):
        fanin_bits = 6 # TODO: Read this from model
        fanout_bits = 2 # TODO: Read this from model
        optimise_with_rarity_string += optimise_with_rarity_template.format(fanin_bits=fanin_bits, fanout_bits=fanout_bits, it="" if i == 0 else i, i=i, path=path, rarity=rarity)
        technology_map_layers_string += technology_map_layer_template.format(fanin_bits=fanin_bits, fanout_bits=fanout_bits, i=i, path=path)
        aig_layers_string += "{path}/aig/layer{i}_opt.aig ".format(i=i, path=path)
        blif_layers_string += "{path}/blif/layer{i}_opt.blif ".format(i=i, path=path)
    gen_monolithic_aig_string = gen_monolithic_aig_template.format(aig_layers_string=aig_layers_string.strip(), path=path)
    gen_monolithic_blif_string = gen_monolithic_blif_template.format(blif_layers_string=blif_layers_string.strip(), path=path)
    return opt_script_template.format(  rarity=rarity,
                                        num_registers=num_registers,
                                        path=path,
                                        optimise_with_rarity_string=optimise_with_rarity_string,
                                        gen_monolithic_aig_string=gen_monolithic_aig_string,
                                        technology_map_layers_string=technology_map_layers_string,
                                        gen_monolithic_blif_string=gen_monolithic_blif_string)


