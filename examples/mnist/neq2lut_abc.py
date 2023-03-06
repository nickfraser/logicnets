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
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from logicnets.nn import    generate_truth_tables, \
                            lut_inference, \
                            module_list_to_verilog_module, \
                            load_histograms
from logicnets.synthesis import synthesize_and_get_resource_counts_with_abc

from models import MnistNeqModel, MnistLutModel
from train import configs, model_config, test
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
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="mnist-s",
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
    parser.add_argument('--input-dropout', type=float, default=None,
        help="The amount of dropout to apply at the model input (default: %(default)s)")
    parser.add_argument('--clock-period', type=float, default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)")
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
    options_cfg = {}
    for k in other_options.keys():
        if k == 'cuda':
            continue
        options_cfg[k] = config[k]

    trans = transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(partial(torch.reshape, shape=(-1,)))
            ])

    # Fetch the test set
    dataset = {}
    dataset["train"] = MNIST('./data', train=True, download=True, transform=trans)
    dataset["test"] = MNIST('./data', train=False, download=True, transform=trans)
    train_loader = DataLoader(dataset["train"], batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=config['batch_size'], shuffle=False)


    # Instantiate the PyTorch model
    x, y = dataset[args.dataset_split][0]
    model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = 10
    model = MnistNeqModel(model_cfg)

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
    lut_model = MnistLutModel(model_cfg)
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
    print("LUT-Based AVG ROC AUC: %f" % (lut_avg_roc_auc))
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

