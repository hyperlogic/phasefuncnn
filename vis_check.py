#
# loads in a bvh file and displays it's animaiton
# also loads in the cooked input and output tensor and displays it.
# they SHOULD match.
#

import os
import sys

import torch
from bvh import Bvh
from wgpu.gui.auto import WgpuCanvas, run

INPUT_DIR = "../PFNN/data/animations/"
OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4

class RenderBuddy:
    def __init__(bvh):
        pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)
    inbasepath = os.path.join(INPUT_DIR, mocap_basename)

    bvh_filename = inbasepath + ".bvh"
    print(f"Loading {bvh_filename}")
    with open(bvh_filename) as f:
        bvh = Bvh(f.read())

    renderBuddy = RenderBuddy(bvh)
    run()
