#!/usr/bin/env python
"""
Single entrypoint for MGHD training. CUDA-Q trajectories are the default sampler.

Usage (placeholder):
  python -m tools.train_core --sampler cudaq --shots 1000
"""
import argparse
from samplers import get_sampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", default="cudaq", choices=["cudaq", "stim"])
    p.add_argument("--shots", type=int, default=128)
    args = p.parse_args()

    sampler = get_sampler(args.sampler)  # defaults to cudaq
    print(f"[train_core] sampler={args.sampler} shots={args.shots}")
    # In Step B/C we will fetch a code from codes_registry and actually train.


if __name__ == "__main__":
    main()
