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

from train import configs, model_config, dataset_config, test
from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel

other_options = {
    "checkpoint": None,
    "input_verilog": None,
    "num_registers": None,
}

if __name__ == "__main__":
    parser = ArgumentParser(description="Synthesize convert a PyTorch trained model into verilog")
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
    parser.add_argument('--dataset-config', type=str, default='config/yaml_IP_OP_config.yml',
        help="The file to use to configure the input dataset (default: %(default)s)")
    parser.add_argument('--dataset-split', type=str, default='test', choices=['train', 'test'],
        help="Dataset to use for evaluation (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="The checkpoint file which contains the model weights")
    parser.add_argument('--input-verilog', type=str, required=True,
        help="The input verilog file to simulate")
    parser.add_argument('--num-registers', type=int, default=0,
        help="The number of pipeline registers in the verilog (default: %(default)s)")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

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
    dataset[args.dataset_split] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split=args.dataset_split)
    test_loader = DataLoader(dataset[args.dataset_split], batch_size=config['batch_size'], shuffle=False)

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
    baseline_accuracy, baseline_avg_roc_auc = test(model, test_loader, cuda=False)
    print("Baseline accuracy: %f" % (baseline_accuracy))
    print("Baseline AVG ROC AUC: %f" % (baseline_avg_roc_auc))

    verilog_dir = os.path.dirname(options_cfg["input_verilog"])
    filename = os.path.split(options_cfg["input_verilog"])[-1]
    print(f"Running inference simulation of Verilog-based model ({filename})")
    model.verilog_inference(verilog_dir, filename, logfile=None, add_registers=options_cfg["num_registers"] != 0, verify=False)
    model.latency = options_cfg["num_registers"]
    verilog_accuracy, verilog_avg_roc_auc = test(model, test_loader, cuda=False)
    print("Verilog-Based Model accuracy: %f" % (verilog_accuracy))
    print("Verilog-Based AVG ROC AUC: %f" % (verilog_avg_roc_auc))

