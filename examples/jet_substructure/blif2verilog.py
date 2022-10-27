#  Copyright (C) 2022 Xilinx, Inc
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

# A script to convert technology-mapped BLIF files to technology mapped verilog.

import os
import glob
import shutil
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from logicnets.abc import   tech_map_to_verilog,\
                            putontop_blif,\
                            pipeline_tech_mapped_circuit
from logicnets.verilog import   fix_abc_module_name,\
                                generate_abc_verilog_wrapper

from train import configs, model_config, dataset_config, test
from models import JetSubstructureNeqModel
from dataset import JetSubstructureDataset

other_options = {
    "output_directory": None,
    "input_blifs": None,
    "num_registers": None,
    "generated_module_name_prefix": None,
}

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert a technology-mapped BLIF files into a technology-mapped verilog file, using ABC")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="jsc-s",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--input-bitwidth', type=int, default=None,
        help="Bitwidth to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-bitwidth', type=int, default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)")
    parser.add_argument('--output-bitwidth', type=int, default=None,
        help="Bitwidth to use at the output (default: %(default)s)")
    parser.add_argument('--input-fanin', type=int, default=None,
        help="Fanin to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-fanin', type=int, default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)")
    parser.add_argument('--output-fanin', type=int, default=None,
        help="Fanin to use at the output (default: %(default)s)")
    parser.add_argument('--hidden-layers', nargs='+', type=int, default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)")
    parser.add_argument('--dataset-file', type=str, default='data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--dataset-config', type=str, default='config/yaml_IP_OP_config.yml',
        help="The file to use to configure the input dataset (default: %(default)s)")
    parser.add_argument('--dataset-split', type=str, default='test', choices=['train', 'test'],
        help="Dataset to use for evaluation (default: %(default)s)")
    parser.add_argument('--input-blifs', nargs='+', type=str, required=True,
        help="The input BLIF files")
    parser.add_argument('--output-directory', type=str, default='./log',
        help="The directory which the generated verilog gets stored. (default: %(default)s)")
    parser.add_argument('--num-registers', type=int, default=0,
        help="The number of registers to add to the generated verilog (default: %(default)s)")
    parser.add_argument('--generated-module-name-prefix', type=str, default='\\aig',
        help="A prefix which matches the module name in the generated verilog, but no other line of code (default: %(default)s)")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]

    # Fetch the test set
    dataset = {}
    dataset[args.dataset_split] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split=args.dataset_split)
    test_loader = DataLoader(dataset[args.dataset_split], batch_size=1, shuffle=False)

    # Instantiate the PyTorch model
    x, y = dataset[args.dataset_split][0]
    model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = len(y)
    model = JetSubstructureNeqModel(model_cfg)

    abc_project_root = options_cfg['output_directory']
    veropt_dir = options_cfg['output_directory']
    input_blif = "layers_full_opt.blif"
    verbose = False

    if len(options_cfg['input_blifs']) > 1:
        nodes, out, err = putontop_blif([os.path.realpath(blif) for blif in options_cfg['input_blifs']], input_blif, working_dir=abc_project_root, verbose=verbose)
    else:
        shutil.copy(os.path.realpath(options_cfg['input_blifs'][0]), f"{abc_project_root}/{input_blif}")

    if options_cfg['num_registers'] == 0:
        nodes, out, err = tech_map_to_verilog(circuit_file=input_blif, output_verilog=f"layers_full_opt.v", working_dir=abc_project_root, verbose=verbose)
    else:
        nodes, out, err = pipeline_tech_mapped_circuit(circuit_file=input_blif, output_verilog=f"layers_full_opt.v", num_registers=options_cfg['num_registers'], working_dir=abc_project_root, verbose=verbose)

    # Fix the resultant verilog file so that it can be simulated
    fix_abc_module_name(f"{veropt_dir}/layers_full_opt.v", f"{veropt_dir}/layers_full_opt.v", options_cfg["generated_module_name_prefix"], "layers_full_opt", add_timescale=True)

    # Generate top-level entity wrapper
    module_list = model.module_list
    _, input_bitwidth = module_list[0].input_quant.get_scale_factor_bits()
    _, output_bitwidth = module_list[-1].output_quant.get_scale_factor_bits()
    input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
    total_input_bits = module_list[0].in_features*input_bitwidth
    total_output_bits = module_list[-1].out_features*output_bitwidth
    module_name="logicnet"
    veropt_wrapper_str = generate_abc_verilog_wrapper(module_name=module_name, input_name="M0", input_bits=total_input_bits, output_name=f"M{len(module_list)}", output_bits=total_output_bits, submodule_name="layers_full_opt", num_registers=options_cfg['num_registers'])
    with open(f"{veropt_dir}/{module_name}.v", "w") as f:
        f.write(veropt_wrapper_str)

    print(f"Adding Nitro-Parts-Lib to {veropt_dir}")
    source_files = glob.glob(f"{os.environ['NITROPARTSLIB']}/*.v")
    for f in source_files:
        shutil.copy(f, f"{veropt_dir}")

