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

from argparse import ArgumentParser

from logicnets.synthesis import synthesize_and_get_resource_counts

if __name__ == "__main__":
    parser = ArgumentParser(description="Synthesize convert a PyTorch trained model into verilog")
    parser.add_argument('--log-dir', type=str, default='./log', required=True,
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--clock-period', type=float, default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)")
    args = parser.parse_args()
    print("Running out-of-context synthesis")
    ret = synthesize_and_get_resource_counts(args.log_dir, "logicnet", fpga_part="xcu280-fsvh2892-2L-e", clk_period_ns=args.clock_period, post_synthesis = 1)

