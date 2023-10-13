#!/usr/bin/env python3

import argparse

from tool.darknet2pytorch import convert


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--cfg", type=str, required=True,
        help="Darknet model.cfg file."
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=True,
        help="Darknet .weights file."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output .torchscript file."
    )
    parser.add_argument(
        "--for-inference", action="store_true",
        help="Set up if you convert the model for inference only."
    )

    args = parser.parse_args()

    convert(args.cfg, args.weights, args.output, args.for_inference)
