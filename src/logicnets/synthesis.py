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

import os
import subprocess
import shutil
from shutil import which
import glob

from .abc import    verilog_bench_to_aig,\
                    txt_to_sim,\
                    simulate_circuit,\
                    putontop_aig,\
                    putontop_blif,\
                    optimize_bdd_network,\
                    evaluate_accuracy,\
                    tech_map_circuit,\
                    iterative_mfs2_optimize,\
                    pipeline_tech_mapped_circuit,\
                    tech_map_to_verilog
from .verilog import    generate_abc_verilog_wrapper,\
                        fix_abc_module_name

#xcvu9p-flgb2104-2-i
# TODO: Add option to perform synthesis on a remote server
# Synthesise design with vivado and get resource counts
def synthesize_and_get_resource_counts(verilog_dir, top_name, fpga_part = "xcku3p-ffva676-1-e", clk_name="clk", clk_period_ns=5.0, post_synthesis = 0):
    # old part : "xczu3eg-sbva484-1-i"
    # ensure that the OH_MY_XILINX envvar is set
    if "OHMYXILINX" not in os.environ:
        raise Exception("The environment variable OHMYXILINX is not defined.")
    # ensure that vivado is in PATH: source $VIVADO_PATH/settings64.sh
    if which("vivado") is None:
        raise Exception("vivado is not in PATH, ensure settings64.sh is sourced.")
    omx_path = os.environ["OHMYXILINX"]
    script = "vivadocompile.sh"
    # vivadocompile.sh <top-level-entity> <clock-name (optional)> <fpga-part (optional)>
    call_omx = "zsh %s/%s %s %s %s %f %s" % (omx_path, script, top_name, clk_name, fpga_part, float(clk_period_ns), post_synthesis)
    call_omx = call_omx.split()
    proc = subprocess.Popen(call_omx, cwd=verilog_dir, stdout=subprocess.PIPE, env=os.environ)
    proc.communicate()

    vivado_proj_folder = "%s/results_%s" % (verilog_dir, top_name)
    res_counts_path = vivado_proj_folder + "/res.txt"

    with open(res_counts_path, 'r') as myfile:
        res_data = myfile.read().split("\n")
    ret = {}
    ret["vivado_proj_folder"] = vivado_proj_folder
    for res_line in res_data:
        res_fields = res_line.split("=")
        print(res_fields)
        try:
            ret[res_fields[0]] = float(res_fields[1])
        except ValueError:
            ret[res_fields[0]] = 0
        except IndexError:
            ret[res_fields[0]] = 0
    if ret["WNS"] == 0:
        ret["fmax_mhz"] = 0
    else:
        ret["fmax_mhz"] = 1000.0 / (clk_period_ns - ret["WNS"])
    return ret

# Optimize the design with ABC
def synthesize_and_get_resource_counts_with_abc(verilog_dir, module_list, pipeline_stages=0, freq_thresh=0, train_input_txt="train_input.txt", train_output_txt="train_output.txt", test_input_txt="test_input.txt", test_output_txt="test_output.txt", bdd_opt_cmd="lnetopt", verbose=False):
    if "ABC_ROOT" not in os.environ:
        raise Exception("The environment variable ABC_ROOT is not defined.")
    abc_path = os.environ["ABC_ROOT"]

    # Create directories and symlinks ready for processing with ABC
    project_prefix = "abc"
    abc_project_root = f"{verilog_dir}/{project_prefix}"
    verilog_bench_dir = f"{abc_project_root}/ver"
    aig_dir = f"{abc_project_root}/aig"
    blif_dir = f"{abc_project_root}/blif"
    veropt_dir = f"{abc_project_root}/veropt"
    if not os.path.exists(verilog_bench_dir):
        os.makedirs(verilog_bench_dir)
    if not os.path.exists(aig_dir):
        os.makedirs(aig_dir)
    if not os.path.exists(blif_dir):
        os.makedirs(blif_dir)
    if not os.path.exists(veropt_dir):
        os.makedirs(veropt_dir)
    # Fetch the right source files from the verilog directory
    source_files = glob.glob(f"{verilog_dir}/logicnet.v") + [f"{verilog_dir}/layer{i}.v" for i in range(len(module_list))] + glob.glob(f"{verilog_dir}/*.bench")
    for f in source_files:
        shutil.copy(f, verilog_bench_dir)
    # Fetch the I/O files
    for f in list(map(lambda x: f"{verilog_dir}/{x}", [train_input_txt, train_output_txt, test_input_txt, test_output_txt])):
        shutil.copy(f, f"{abc_project_root}")

    # Preparation - model / I/O conversion
    # Convert txt inputs into the sim format
    out, err = txt_to_sim(train_input_txt, "train.sim", working_dir=abc_project_root, verbose=verbose)
    out, err = txt_to_sim(test_input_txt, "test.sim", working_dir=abc_project_root)

    # Create AIGs from verilog
    for i in range(len(module_list)):
        nodes, out, err = verilog_bench_to_aig(f"ver/layer{i}.v", f"aig/layer{i}.aig", working_dir=abc_project_root, verbose=verbose)

    # Simulate each layer
    for i in range(len(module_list)):
        out, err = simulate_circuit(f"aig/layer{i}.aig", f"train{i}.sim" if i != 0 else "train.sim", f"train{i+1}.sim", working_dir=abc_project_root, verbose=verbose)

    # Synthesis
    average_tt_pcts = []
    for i in range(len(module_list)):
        _, input_bitwidth = module_list[i].input_quant.get_scale_factor_bits()
        _, output_bitwidth = module_list[i].output_quant.get_scale_factor_bits()
        indices, _, _, _ = module_list[i].neuron_truth_tables[0]
        fanin = len(indices)
        nodes, tt_pct, time, out, err = optimize_bdd_network(f"aig/layer{i}.aig", f"aig/layer{i}_full.aig", int(input_bitwidth*fanin), int(output_bitwidth), freq_thresh, f"train{i}.sim" if i != 0 else "train.sim", opt_cmd=bdd_opt_cmd, working_dir=abc_project_root, verbose=verbose)
        average_tt_pcts.append(tt_pct)

    # Technology mapping
    for i in range(len(module_list)):
        _, input_bitwidth = module_list[i].input_quant.get_scale_factor_bits()
        _, output_bitwidth = module_list[i].output_quant.get_scale_factor_bits()
        indices, _, _, _ = module_list[i].neuron_truth_tables[0]
        fanin = len(indices)
        out, err = tech_map_circuit(f"aig/layer{i}_full.aig", f"blif/layer{i}_full.blif", int(input_bitwidth*fanin), int(output_bitwidth), working_dir=abc_project_root, verbose=verbose)

    # Generate monolithic circuits
    if len(module_list) > 1:
        nodes, out, err = putontop_aig([f"aig/layer{i}_full.aig" for i in range(len(module_list))], f"aig/layers_full.aig", working_dir=abc_project_root, verbose=verbose)
        nodes, out, err = putontop_blif([f"blif/layer{i}_full.blif" for i in range(len(module_list))], f"blif/layers_full.blif", working_dir=abc_project_root, verbose=verbose)
    else:
        shutil.copy(f"{aig_dir}/layer0_full.aig", f"{aig_dir}/layers_full.aig")
        shutil.copy(f"{blif_dir}/layer0_full.blif", f"{blif_dir}/layers_full.blif")

    # Generic logic synthesis optimizations
    nodes = iterative_mfs2_optimize(circuit_file=f"blif/layers_full.blif", output_file=f"blif/layers_full_opt.blif", tmp_file="blif/tmp.blif", max_loop=100, working_dir=abc_project_root, verbose=verbose)

    # Generate verilog, with or without pipelining
    if pipeline_stages == 0:
        nodes, out, err = tech_map_to_verilog(circuit_file=f"blif/layers_full_opt.blif", output_verilog=f"veropt/layers_full_opt.v", working_dir=abc_project_root, verbose=verbose)
    else:
        nodes, out, err = pipeline_tech_mapped_circuit(circuit_file=f"blif/layers_full_opt.blif", output_verilog=f"veropt/layers_full_opt.v", num_registers=num_registers, working_dir=abc_project_root, verbose=verbose)
    fix_abc_module_name(f"{veropt_dir}/layers_full_opt.v", f"{veropt_dir}/layers_full_opt.v", "\\aig", "layers_full_opt", add_timescale=True)

    # Generate top-level entity wrapper
    _, input_bitwidth = module_list[0].input_quant.get_scale_factor_bits()
    _, output_bitwidth = module_list[-1].output_quant.get_scale_factor_bits()
    input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
    total_input_bits = module_list[0].in_features*input_bitwidth
    total_output_bits = module_list[-1].out_features*output_bitwidth
    module_name="logicnet"
    veropt_wrapper_str = generate_abc_verilog_wrapper(module_name=module_name, input_name="M0", input_bits=total_input_bits, output_name=f"M{len(module_list)}", output_bits=total_output_bits, submodule_name="layers_full_opt", num_registers=pipeline_stages)
    with open(f"{veropt_dir}/{module_name}.v", "w") as f:
        f.write(veropt_wrapper_str)

    # Evaluation
    # Training set:
    _, output_bitwidth = module_list[-1].output_quant.get_scale_factor_bits()
    out, err = simulate_circuit(f"blif/layers_full_opt.blif", "train.sim", "train.simo", working_dir=abc_project_root, verbose=verbose)
    train_accuracy, out, err = evaluate_accuracy(f"blif/layers_full_opt.blif", "train.simo", train_output_txt, int(output_bitwidth), working_dir=abc_project_root, verbose=verbose)
    # Test set:
    out, err = simulate_circuit(f"blif/layers_full_opt.blif", "test.sim", "test.simo", working_dir=abc_project_root, verbose=verbose)
    test_accuracy, out, err = evaluate_accuracy(f"blif/layers_full_opt.blif", "test.simo", test_output_txt, int(output_bitwidth), working_dir=abc_project_root, verbose=verbose)

    return train_accuracy, test_accuracy, nodes, average_tt_pcts

