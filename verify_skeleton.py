#
# Copyright (c) 2025 Anthony J. Thibault
# This software is licensed under the MIT License. See LICENSE for more details.
#
# go thru the mocap data and ensure
# verify that all the bvh have the same skeleton

import os
import sys

import numpy as np

import mocap

OUTPUT_DIR = "output"

mocap_paths = [
    "../PFNN/data/animations/LocomotionFlat01_000.bvh",
    "../PFNN/data/animations/LocomotionFlat02_000.bvh",
    "../PFNN/data/animations/LocomotionFlat02_001.bvh",
    "../PFNN/data/animations/LocomotionFlat03_000.bvh",
    "../PFNN/data/animations/LocomotionFlat04_000.bvh",
    "../PFNN/data/animations/LocomotionFlat05_000.bvh",
    "../PFNN/data/animations/LocomotionFlat06_000.bvh",
    "../PFNN/data/animations/LocomotionFlat06_001.bvh",
    "../PFNN/data/animations/LocomotionFlat07_000.bvh",
    "../PFNN/data/animations/LocomotionFlat08_000.bvh",
    "../PFNN/data/animations/LocomotionFlat08_001.bvh",
    "../PFNN/data/animations/LocomotionFlat09_000.bvh",
    "../PFNN/data/animations/LocomotionFlat10_000.bvh",
    "../PFNN/data/animations/LocomotionFlat11_000.bvh",
    "../PFNN/data/animations/LocomotionFlat12_000.bvh",
]

mocap_names = [os.path.splitext(os.path.basename(n))[0] for n in mocap_paths]

if __name__ == "__main__":

    skeletons = []
    for mocap_name in mocap_names:
        outbasepath = os.path.join(OUTPUT_DIR, mocap_name)
        skeletons.append(mocap.unpickle_obj(outbasepath + "_skeleton.pkl"))

    # compare skeleton to reference
    ref_skeleton = skeletons[0]
    for skeleton in skeletons[1:]:
        assert len(ref_skeleton.joint_names) == len(skeleton.joint_names)
        for i in range(len(skeleton.joint_names)):
            assert ref_skeleton.joint_names[i] == skeleton.joint_names[i]
            name = skeleton.joint_names[i]
            assert ref_skeleton.get_joint_offset(name) == skeleton.get_joint_offset(name)

    for name in ref_skeleton.joint_names:
        offset = np.array(ref_skeleton.get_joint_offset(name))
        print(f"{name}: {np.linalg.norm(offset)}")

    height = ref_skeleton.get_joint_offset('Head')[1]
    print(f"height = {height}")

