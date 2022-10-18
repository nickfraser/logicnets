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

import os
import subprocess
import re
import shutil

_aig_re_str = r'and\s+=\s+\d+'
_lut_re_str = r'nd\s+=\s+\d+'
_acc_re_str = r'The\s+accuracy\s+is\s+\d+\.\d+'
_avg_cs_re_str = r'Average\s+care\s+set\s+is\s+\d+\.\d+'
_elapse_s_re_str = r'elapse:\s+\d+\.\d+'

def verilog_bench_to_aig(verilog_file, aig_file, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"&lnetread {verilog_file}; &ps; &w {aig_file}"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    aig_re = re.compile(_aig_re_str)
    nodes = int(aig_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(nodes)
        print(out)
        print(err)
    return nodes, out, err # TODO: return the number of nodes

def txt_to_sim(txt_file, sim_file, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"&lnetread {txt_file} {sim_file}"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    if verbose:
        print(out)
        print(err)
    return out, err

def simulate_circuit(circuit_file, sim_input_file, sim_output_file, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    if circuit_file.endswith(".aig"):
        cmd = [f"{abc_path}/abc", '-c', f"&r {circuit_file}; &lnetsim {sim_input_file} {sim_output_file}"]
    elif circuit_file.endswith(".blif"):
        cmd = [f"{abc_path}/abc", '-c', f"read {circuit_file}; strash; &get; &lnetsim {sim_input_file} {sim_output_file}"]
    else:
        raise ValueError(f"Unsupported file type: {circuit_file}")
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    if verbose:
        print(out)
        print(err)
    return out, err

def putontop_aig(aig_files, output_aig_file, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"putontop {' '.join(aig_files)}; strash; print_stats; write {output_aig_file}"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    aig_re = re.compile(_aig_re_str)
    nodes = int(aig_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(nodes)
        print(out)
        print(err)
    return nodes, out, err # TODO: return the number of nodes

def putontop_blif(blif_files, output_blif_file, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"putontop {' '.join(blif_files)}; sweep; print_stats; write {output_blif_file}"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    lut_re = re.compile(_lut_re_str)
    nodes = int(lut_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(nodes)
        print(out)
        print(err)
    return nodes, out, err # TODO: return the number of nodes

def optimize_bdd_network(circuit_file, output_file, input_bitwidth, output_bitwidth, rarity, sim_file, opt_cmd="&lnetopt", abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"&r {circuit_file}; &ps; {opt_cmd} -I {input_bitwidth} -O {output_bitwidth} -R {rarity} {sim_file}; &w {output_file}; &ps; time"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    aig_re = re.compile(_aig_re_str)
    nodes = int(aig_re.search(str(out)).group().split(" ")[-1])
    if opt_cmd == "&lnetopt":
        tt_pct_re = re.compile(_avg_cs_re_str)
        tt_pct = float(tt_pct_re.search(str(out)).group().split(" ")[-1])
    else:
        tt_pct = None
    time_re = re.compile(_elapse_s_re_str)
    time_s = float(time_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(nodes)
        print(tt_pct)
        print(time_s)
        print(out)
        print(err)
    return nodes, tt_pct, time_s, out, err # TODO: return the number of nodes, tt%, time

def optimize_mfs2(circuit_file, output_file, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"read {circuit_file}; if -K 6 -a; mfs2; write_blif {output_file}; print_stats"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    lut_re = re.compile(_lut_re_str)
    nodes = int(lut_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(nodes)
        print(out)
        print(err)
    return nodes, out, err # TODO: return the number of nodes

def iterative_mfs2_optimize(circuit_file, output_file, tmp_file="tmp.blif", max_loop=100, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    tmp_file_path = tmp_file if working_dir is None else f"{working_dir}/{tmp_file}"
    output_file_path = output_file if working_dir is None else f"{working_dir}/{output_file}"
    cmd = [f"{abc_path}/abc", '-c', f"read {circuit_file}; sweep; write_blif {tmp_file}; print_stats"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    lut_re = re.compile(_lut_re_str)
    nodes = int(lut_re.search(str(out)).group().split(" ")[-1])
    best = nodes
    shutil.copy(tmp_file_path, output_file_path)
    if verbose:
        print(nodes)
        print(best)
        print(out)
        print(err)
    for i in range(max_loop):
        if i == 0:
            cmd = [f"{abc_path}/abc", '-c', f"read {tmp_file}; mfs2; write_blif {tmp_file}; print_stats"]
            proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
            if verbose:
                print(" ".join(cmd))
            out, err = proc.communicate()
            lut_re = re.compile(_lut_re_str)
            nodes = int(lut_re.search(str(out)).group().split(" ")[-1])
            if verbose:
                print(nodes)
                print(out)
                print(err)
        else:
            nodes, out, err = optimize_mfs2(tmp_file, tmp_file, abc_path=abc_path, working_dir=working_dir, verbose=verbose)
        if nodes >= best:
            break
        else:
            print(best)
            best = nodes
            shutil.copy(tmp_file_path, output_file_path)
    os.remove(tmp_file_path)
    return best

def tech_map_circuit(circuit_file, output_blif, input_bitwidth, output_bitwidth, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"&r {circuit_file}; &lnetmap -I {input_bitwidth} -O {output_bitwidth}; write {output_blif}"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    if verbose:
        print(out)
        print(err)
    return out, err

def pipeline_tech_mapped_circuit(circuit_file, output_verilog, num_registers, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"read {circuit_file}; print_stats; pipe -L {num_registers}; print_stats; retime -M 4; print_stats; sweep; print_stats; write_verilog -fm {output_verilog}"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    lut_re = re.compile(_lut_re_str)
    nodes = int(lut_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(nodes)
        print(out)
        print(err)
    return out, err

def tech_map_to_verilog(circuit_file, output_verilog, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    cmd = [f"{abc_path}/abc", '-c', f"read {circuit_file}; print_stats; write_verilog -fm {output_verilog}"]
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    lut_re = re.compile(_lut_re_str)
    nodes = int(lut_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(nodes)
        print(out)
        print(err)
    return nodes, out, err

def evaluate_accuracy(circuit_file, sim_output_file, reference_txt, output_bitwidth, abc_path=os.environ["ABC_ROOT"], working_dir=None, verbose=False):
    if circuit_file.endswith(".aig"):
        cmd = [f"{abc_path}/abc", '-c', f"&r {circuit_file}; &lneteval -O {output_bitwidth} {sim_output_file} {reference_txt}"]
    elif circuit_file.endswith(".blif"):
        cmd = [f"{abc_path}/abc", '-c', f"read {circuit_file}; strash; &get; &lneteval -O {output_bitwidth} {sim_output_file} {reference_txt}"]
    else:
        raise ValueError(f"Unsupported file type: {circuit_file}")
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()
    acc_re = re.compile(_acc_re_str)
    accuracy = float(acc_re.search(str(out)).group().split(" ")[-1])
    if verbose:
        print(accuracy)
        print(out)
        print(err)
    return accuracy, out, err # TODO: accuracy %, time

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


def generate_opt_script_string(module_list, path, num_registers, rarity=0, opt_cmd="&lnetopt"):
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
    optimise_with_rarity_template = "&r {path}/layer{i}.aig; &ps; {opt_cmd} -I {fanin_bits} -O {fanout_bits} -R {rarity} {path}/train{it}.sim;  &w {path}/aig/layer{i}_opt.aig; &ps; time\n"
    technology_map_layer_template = "&r {path}/aig/layer{i}_opt.aig; &lnetmap -I {fanin_bits} -O {fanout_bits}; write {path}/blif/layer{i}_opt.blif; write_verilog -fm {path}/ver/layer{i}_opt.v\n"
    gen_monolithic_aig_template = "putontop {aig_layers_string}; st; ps; write {path}/aig/layers_opt.aig\n"
    gen_monolithic_blif_template = "putontop {blif_layers_string}; sw; ps; write {path}/blif/layers_opt.blif\n"
    num_layers = len(module_list) # TODO: fetch number of layers from the model
    optimise_with_rarity_string = ""
    technology_map_layers_string = ""
    aig_layers_string = ""
    blif_layers_string = ""
    for i in range(num_layers):
        # Read in fanin/fanout bits
        # Add assertion that fanin/fanout bits for all neuron is that same
        layer = module_list[i]
        _, input_bitwidth = layer.input_quant.get_scale_factor_bits()
        _, output_bitwidth = layer.output_quant.get_scale_factor_bits()
        num_indices = len(layer.neuron_truth_tables[0])
        fanin_bits = input_bitwidth*num_indices
        fanout_bits = output_bitwidth

        # Generate optimisation script.
        optimise_with_rarity_string += optimise_with_rarity_template.format(fanin_bits=fanin_bits, fanout_bits=fanout_bits, it="" if i == 0 else i, i=i, path=path, rarity=rarity, opt_cmd=opt_cmd)
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


