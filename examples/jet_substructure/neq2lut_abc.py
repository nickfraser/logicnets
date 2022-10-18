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
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from logicnets.nn import    generate_truth_tables, \
                            lut_inference, \
                            module_list_to_verilog_module, \
                            load_histograms
from logicnets.synthesis import synthesize_and_get_resource_counts_with_abc

from train import configs, model_config, dataset_config, test
from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel, JetSubstructureLutModel
from dataset_dump import dump_io

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "histograms": None,
    "freq_thresh": None,
}

if __name__ == "__main__":
    parser = ArgumentParser(description="Synthesize convert a PyTorch trained model into verilog using ABC")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="jsc-s",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for evaluation (default: %(default)s)")
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
    parser.add_argument('--clock-period', type=float, default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)")
    parser.add_argument('--dataset-config', type=str, default='config/yaml_IP_OP_config.yml',
        help="The file to use to configure the input dataset (default: %(default)s)")
    parser.add_argument('--dataset-split', type=str, default='test', choices=['train', 'test'],
        help="Dataset to use for evaluation (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="The checkpoint file which contains the model weights")
    parser.add_argument('--histograms', type=str, default=None,
        help="The checkpoint histograms of LUT usage (default: %(default)s)")
    parser.add_argument('--freq-thresh', type=int, default=None,
        help="Threshold to use to include this truth table into the model (default: %(default)s)")
    parser.add_argument('--num-registers', type=int, default=0,
        help="The number of registers to add to the generated verilog (default: %(default)s)")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        if k == 'cuda':
            continue
        options_cfg[k] = config[k]

    # Fetch the test set
    dataset = {}
    dataset["train"] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="train")
    dataset["test"] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="test")
    train_loader = DataLoader(dataset["train"], batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=config['batch_size'], shuffle=False)


    # Instantiate the PyTorch model
    x, y = dataset[args.dataset_split][0]
    model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = len(y)
    model = JetSubstructureNeqModel(model_cfg)

    # Load the model weights
    checkpoint = torch.load(options_cfg['checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint['model_dict'])

    # Test the PyTorch model
    print("Running inference on baseline model...")
    model.eval()
    baseline_accuracy = test(model, test_loader, cuda=False)
    print("Baseline accuracy: %f" % (baseline_accuracy))

    # Run preprocessing on training set.
    #train_input_file = config['log_dir'] + "/train_input.txt"
    #train_output_file = config['log_dir'] + "/train_output.txt"
    #test_input_file = config['log_dir'] + "/test_input.txt"
    #test_output_file = config['log_dir'] + "/test_output.txt"
    #print(f"Dumping train I/O to {train_input_file} and {train_output_file}")
    #dump_io(model, train_loader, train_input_file, train_output_file)
    #print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
    #dump_io(model, test_loader, test_input_file, test_output_file)

    # Instantiate LUT-based model
    lut_model = JetSubstructureLutModel(model_cfg)
    lut_model.load_state_dict(checkpoint['model_dict'])

    # Generate the truth tables in the LUT module
    print("Converting to NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)

    # Test the LUT-based model
    print("Running inference on LUT-based model...")
    lut_inference(lut_model)
    lut_model.eval()
    lut_accuracy = test(lut_model, test_loader, cuda=False)
    print("LUT-Based Model accuracy: %f" % (lut_accuracy))
    modelSave = {   'model_dict': lut_model.state_dict(),
                    'test_accuracy': lut_accuracy}

    torch.save(modelSave, options_cfg["log_dir"] + "/lut_based_model.pth")
    if options_cfg["histograms"] is not None:
        luts = torch.load(options_cfg["histograms"])
        load_histograms(lut_model, luts)

    print("Generating verilog in %s..." % (options_cfg["log_dir"]))
    module_list_to_verilog_module(lut_model.module_list, "logicnet", options_cfg["log_dir"], generate_bench=True, add_registers=False)
    print("Top level entity stored at: %s/logicnet.v ..." % (options_cfg["log_dir"]))

    print("Running synthesis and verilog technology-mapped verilog in ABC")
    train_accuracy, test_accuracy, nodes, average_care_set_size = synthesize_and_get_resource_counts_with_abc(options_cfg["log_dir"], lut_model.module_list, pipeline_stages=args.num_registers, freq_thresh=args.freq_thresh, train_input_txt="train_input.txt", train_output_txt="train_output.txt", test_input_txt="test_input.txt", test_output_txt="test_output.txt", bdd_opt_cmd="&ttopt", verbose=False)
    print(f"Training set accuracy(%): {train_accuracy}")
    print(f"Test set accuracy(%): {test_accuracy}")
    print(f"LUT6(#): {nodes}")
    print(f"Average care set sizes(%): {average_care_set_size}")

