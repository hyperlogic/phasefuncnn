import glm
import mocap
import os
import pickle
import sys

OUTPUT_DIR = "output"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]

    # unpickle skeleton
    skeleton_filename = os.path.join(OUTPUT_DIR, mocap_basename + "_skeleton.pkl")
    with open(skeleton_filename, "rb") as f:
        skeleton = pickle.load(f)

    # unpickle xforms
    xforms_filename = os.path.join(OUTPUT_DIR, mocap_basename + "_xforms.pkl")
    with open(xforms_filename, "rb") as f:
        xforms = pickle.load(f)

    vels = mocap.build_vels(skeleton, xforms)

    # pickle vels
    vels_filename = os.path.join(OUTPUT_DIR, mocap_basename + "_vels.pkl")
    with open(vels_filename, "wb") as f:
        xforms = pickle.dump(vels, f)
