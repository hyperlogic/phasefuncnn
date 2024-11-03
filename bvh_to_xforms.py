from bvh import Bvh
import glm
import mocap
import os
import pickle
import sys

OUTPUT_DIR = "output"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected bvh file argument")
        exit(1)

    mocap_filename = sys.argv[1]
    mocap_basename = os.path.splitext(os.path.basename(mocap_filename))[0]

    print(f"Loading {mocap_filename}")
    with open(mocap_filename) as f:
        bvh = Bvh(f.read())

    skeleton = mocap.Skeleton(bvh)
    print(skeleton.joint_names)

    xforms = mocap.build_xforms(bvh, skeleton)

    # create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pickle skeleton
    skeleton_filename = os.path.join(OUTPUT_DIR, mocap_basename + "_skeleton.pkl")
    with open(skeleton_filename, "wb") as f:
        pickle.dump(skeleton, f)

    # pickle xforms
    xforms_filename = os.path.join(OUTPUT_DIR, mocap_basename + "_xforms.pkl")
    with open(xforms_filename, "wb") as f:
        pickle.dump(xforms, f)
