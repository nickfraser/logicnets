# LogicNets for Jet-Substructure Classification

This example shows the accuracy that is attainable using the LogicNets methodology on the jet substructure classification task described in our 2020 FPL paper.
This example is a reimplementation of that work.

## Prerequisites

* LogicNets
* h5py
* yaml<6.0
* numpy
* pandas
* scikit-learn

## Installation

If you're using the docker image, all the above prerequisites will be already installed.
Otherwise, you can install the above dependencies with pip and/or conda.

## Download the Dataset

In order to download the dataset, browse to the directory where this example is contained (e.g., `cd /path/to/logicnets/examples/jet_substructure/`) and run the following:

```bash
mkdir -p data
wget https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z
```

## Usage

To train the \"JSC-S\", \"JSC-M\" and \"JSC-L\" networks described in our 2020 FPL paper, run the
following:

```bash
python train.py --arch <jsc-s|jsc-m|jsc-l> --log-dir ./<jsc_s|jsc_m|jsc_l>/
```

To then generate verilog from this trained model, run the following:

```bash
python neq2lut.py --arch <jsc-s|jsc-m|jsc-l> --checkpoint ./<jsc_s|jsc_m|jsc_l>/best_accuracy.pth --log-dir ./<jsc_s|jsc_m|jsc_l>/verilog/ --add-registers
```

## Results

Your results may vary slightly, depending on your system configuration.
The following results are attained when training on a CPU and synthesising with Vivado 2019.2:

| Network Architecture  | Test Accuracy (%) | LUTs  | Flip Flops    | Fmax (Mhz)    | Latency (Cycles)  |
| --------------------- | ----------------- | ----- | ------------- | ------------- | ----------------- |
| JSC-S                 |              69.8 |   244 |           270 |       1353.18 |                 5 |
| JSC-M                 |              72.1 | 15526 |           881 |        576.70 |                 5 |
| JSC-L                 |              73.1 | 36415 |          2790 |        389.86 |                 6 |

Note, the model architectures reflect the architectures described in our [FPL'20 paper](https://arxiv.org/abs/2004.03021).

## Citation

If you find this work useful for your research, please consider citing
our paper below:

```bibtex
@inproceedings{umuroglu2020logicnets,
  author = {Umuroglu, Yaman and Akhauri, Yash and Fraser, Nicholas J and Blott, Michaela},
  booktitle = {Proceedings of the International Conference on Field-Programmable Logic and Applications},
  title = {LogicNets: Co-Designed Neural Networks and Circuits for Extreme-Throughput Applications},
  year = {2020},
  pages = {291-297},
  publisher = {IEEE Computer Society},
  address = {Los Alamitos, CA, USA},
  month = {sep}
}
```

## Testing BLIF Files on the JSC Dataset

In this section, we show how to take technology-mapped BLIF files,
generate technology-mapped verilog and simulate the verilog on the JSC dataset.

### Convert BLIF Files into Verilog

To convert the full BLIF files (as generated from the LogicNets examples, via `neq2lut_abc.py`) into verilog, run the following:

```bash
python blif2verilog.py --arch <jsc-s|jsc-m|jsc-l> --input-blif <path_to_tech_mapped_blif>/layers_full_opt.blif --output-directory <output_directory>
```

To convert the layer-wise BLIF files into verilog, run the following:

```bash
python blif2verilog.py --arch <jsc-s|jsc-m|jsc-l> --input-blifs <path_to_tech_mapped_blif>/*.blif --output-directory <output_directory> --generated-module-name-prefix layer0
```

Note, the generated module name prefix will likely have to change if the source files are handled in a different way.

### Simulate Verilog

The resultant verilog can be simulated as follows:

```bash
python simulate_verilog.py --arch <jsc-s|jsc-m|jsc-l> --checkpoint <path_to_checkpoint> --input-verilog <output_directory>/logicnet.v
```

