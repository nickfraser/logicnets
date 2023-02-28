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
                            save_luts, \
                            module_list_to_verilog_module

from models import MnistNeqModel, MnistLutModel
from train import configs, model_config, other_options, test
from logicnets.synthesis import synthesize_and_get_resource_counts

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate histograms of states used throughout LogicNets")
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
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the calculated histograms (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="The checkpoint file which contains the model weights")
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
    dataset[args.dataset_split] = MNIST('./data', train=args.dataset_split == "train", download=True, transform=trans)
    test_loader = DataLoader(dataset[args.dataset_split], batch_size=config['batch_size'], shuffle=False)

    # Instantiate the PyTorch model
    x, y = dataset[args.dataset_split][0]
    model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = 10
    model = MnistNeqModel(model_cfg)

    # Load the model weights
    checkpoint = torch.load(options_cfg['checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint['model_dict'])

    # Test the PyTorch model
    print("Running inference of baseline model on training set (%d examples)..." % (dataset_length))
    model.eval()
    baseline_accuracy = test(model, train_loader, cuda=False)
    print("Baseline accuracy: %f" % (baseline_accuracy))

    # Instantiate LUT-based model
    lut_model = MnistLutModel(model_cfg)
    lut_model.load_state_dict(checkpoint['model_dict'])

    # Generate the truth tables in the LUT module
    print("Converting to NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)

    # Test the LUT-based model
    print("Running inference of LUT-based model training set (%d examples)..." % (dataset_length))
    lut_inference(lut_model, track_used_luts=True)
    lut_model.eval()
    lut_accuracy = test(lut_model, train_loader, cuda=False)
    print("LUT-Based Model accuracy: %f" % (lut_accuracy))
    print("Saving LUTs to %s... " % (options_cfg["log_dir"] + "/luts.pth"))
    save_luts(lut_model, options_cfg["log_dir"] + "/luts.pth")
    print("Done!")

