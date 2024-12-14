import mocap
import numpy as np
import os
import pickle
import pygfx as gfx
import torch
from typing import Tuple
from tqdm import trange, tqdm
import sys
from wgpu.gui.auto import WgpuCanvas, run

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4



def build_indices(start: int, stride: int, repeat: int = 1) -> Tuple[int, list[int]]:
    indices = [i * stride + start for i in range(repeat)]
    offset = repeat * stride + start
    return offset, indices

def build_input_map(skeleton: mocap.Skeleton) -> dict[str, [int]]:
    num_joints = skeleton.num_joints

    input_map = {}
    offset = 0
    next_offset, indices = build_indices(offset, 4, TRAJ_WINDOW_SIZE)
    input_map["traj_pos_i"] = {"size": 2, "indices": indices}
    _, indices = build_indices(offset + 2, 4, TRAJ_WINDOW_SIZE)
    input_map["traj_vel_i"] = {"size": 2, "indices": indices}

    offset = next_offset
    next_offset, indices = build_indices(offset, 6, num_joints)
    input_map["joint_pos_i-1"] = {"size": 3, "indices": indices}
    _, indices = build_indices(offset + 3, 6, num_joints)
    input_map["joint_vel_i-1"] = {"size": 3, "indices": indices}

    for k, v in input_map.items():
        print(f"{k}: {v}")

    return input_map

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle/load data
    skeleton = mocap.unpickle_obj(outbasepath + "_skeleton.pkl")
    input_map = build_input_map(skeleton)

    print(f"skeleton.num_joints = {skeleton.num_joints}")
    X = torch.load(os.path.join(OUTPUT_DIR, "X.pth"), weights_only=True)
    print(f"X.shape = {X.shape}")

    num_cols = input_map["joint_vel_i-1"]["indices"][-1] + input_map["joint_vel_i-1"]["size"]
    assert num_cols == X.shape[1]

