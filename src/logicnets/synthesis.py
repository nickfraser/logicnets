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

from .abc import generate_prepare_script_string, generate_opt_script_string

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
def synthesize_and_get_resource_counts_with_abc(verilog_dir, module_list, pipeline_stages=0, freq_thresh=0):
    if "ABC_ROOT" not in os.environ:
        raise Exception("The environment variable ABC_ROOT is not defined.")
    abc_path = os.environ["ABC_ROOT"]

    # Create directories and symlinks ready for processing with ABC
    project_prefix = "logicnet"
    abc_project_root = f"{verilog_dir}/{project_prefix}"
    verilog_bench_dir = f"{abc_project_root}/ver"
    aig_dir = f"{abc_project_root}/aig"
    blif_dir = f"{abc_project_root}/blif"
    if not os.path.exists(verilog_bench_dir):
        os.makedirs(verilog_bench_dir)
    if not os.path.exists(aig_dir):
        os.makedirs(aig_dir)
    if not os.path.exists(blif_dir):
        os.makedirs(blif_dir)
    real_abc_project_root = os.path.realpath(abc_project_root)
    project_symlink_path = f"{abc_path}/{project_prefix}"
    os.symlink(real_abc_project_root, project_symlink_path) # Create a symlink to this folder in the ABC root.
    # Fetch the right source files from the verilog directory
    source_files = glob.glob(f"{verilog_dir}/*.v") + glob.glob(f"{verilog_dir}/*.bench")
    for f in source_files:
        shutil.copy(f, verilog_bench_dir)
    # Fetch the I/O files
    for f in glob.glob(f"{verilog_dir}/*.txt"):
        shutil.copy(f, f"{abc_project_root}")

    # Create script files to pass to ABC
    # TODO: Calculate number of layers from the model
    with open(f"{abc_project_root}/prepare.script", "w") as f:
        f.write(generate_prepare_script_string(num_layers=len(module_list), path=project_prefix))
    with open(f"{abc_project_root}/opt_all.script", "w") as f:
        f.write(generate_opt_script_string(module_list=module_list, path=project_prefix, num_registers=pipeline_stages, rarity=freq_thresh))

    #proc = subprocess.Popen(['./abc', '-c', '"x/jsc_s/prepare.script"', '-c', '"x/jsc_s/opt_all.script"'], cwd=abc_path, stdout=subprocess.PIPE, env=os.environ)
    proc = subprocess.Popen(['./abc', '-c', f'"{project_prefix}/prepare.script"', '-c', f'"{project_prefix}/opt_all.script"'], cwd=abc_path, stdout=subprocess.PIPE, env=os.environ)
    out, err = proc.communicate()

    with open(f"{abc_project_root}/abc.log", "w") as f:
        f.write(out.decode("utf-8"))

    os.remove(project_symlink_path)

