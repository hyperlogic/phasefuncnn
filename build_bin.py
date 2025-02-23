import glob
import os
import pickle
import sys

import numpy as np
from build_traj import TRAJ_SAMPLE_RATE
from inference import NUM_GAITS, SAMPLE_RATE
import torch

import datalens
from pfnn import NUM_CONTROL_POINTS, PFNN

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12

def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":

    weights_filename = os.path.join(OUTPUT_DIR, "final_checkpoint.pth")
    if len(sys.argv) > 1:
        weights_filename = sys.argv[1]

    torch.no_grad()

    device = torch.device("cpu")

    # unpickle skeleton
    # pick ANY skeleton in the output dir, they should all be the same.
    skeleton_files = glob.glob(os.path.join(OUTPUT_DIR, "skeleton/*.pkl"))
    assert len(skeleton_files) > 0, "could not find any pickled skeletons in output folder"
    skeleton = unpickle_obj(skeleton_files[0])

    x_lens = datalens.InputLens(TRAJ_WINDOW_SIZE, skeleton.num_joints)
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, skeleton.num_joints)

    # load model
    in_features = x_lens.num_cols
    out_features = y_lens.num_cols
    print(f"PFNN(in_features = {in_features}, out_features = {out_features}, device = {device}")
    model = PFNN(in_features, out_features, device=device)
    model.eval()  # deactivate dropout
    state_dict = torch.load(weights_filename, weights_only=True, map_location=device)

    # load input mean, std and weights. used to unnormalize the inputs
    X = torch.load(os.path.join(OUTPUT_DIR, "X.pth"), weights_only=True, map_location=device)
    X_mean = torch.load(os.path.join(OUTPUT_DIR, "X_mean.pth"), weights_only=True, map_location=device)
    X_std = torch.load(os.path.join(OUTPUT_DIR, "X_std.pth"), weights_only=True, map_location=device)
    X_w = torch.load(os.path.join(OUTPUT_DIR, "X_w.pth"), weights_only=True, map_location=device)

    # load output mean and std. used to unnormalize the outputs
    Y = torch.load(os.path.join(OUTPUT_DIR, "Y.pth"), weights_only=True, map_location=device)
    Y_mean = torch.load(os.path.join(OUTPUT_DIR, "Y_mean.pth"), weights_only=True, map_location=device)
    Y_std = torch.load(os.path.join(OUTPUT_DIR, "Y_std.pth"), weights_only=True, map_location=device)

    P = torch.load(os.path.join(OUTPUT_DIR, "P.pth"), weights_only=True, map_location=device)

    model.load_state_dict(state_dict)

    # dump model params into a binary file
    bin_filename = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(weights_filename))[0] + ".bin")
    with open(bin_filename, "wb") as out:
        with torch.no_grad():
            print("// generated from build_bin.py")

            print(f"constexpr size_t NUM_JOINTS = {skeleton.num_joints};")
            print(f"constexpr size_t TRAJ_WINDOW_SIZE = {TRAJ_WINDOW_SIZE};")
            print(f"constexpr size_t INPUT_SIZE = {in_features};")
            print(f"constexpr size_t OUTPUT_SIZE = {out_features};")
            print(f"constexpr size_t NUM_CONTROL_POINTS = {NUM_CONTROL_POINTS};")
            print(f"constexpr size_t NUM_GAITS = {NUM_GAITS};")
            print(f"constexpr size_t SAMPLE_RATE = {SAMPLE_RATE};")
            print(f"constexpr size_t TRAJ_SAMPLE_RATE = {TRAJ_SAMPLE_RATE};")
            for name, param in model.named_parameters():
                print(
                    f"constexpr std::array<size_t, {len(param.shape)}> {name.replace('.', '_').upper()}_SIZE = {{{', '.join([str(x) for x in param.shape])}}};"
                )

            print("struct Params {")
            for name, param in model.named_parameters():
                flat = param.cpu().flatten()
                print(f"    float {name.replace('.', '_')}[{len(flat)}];")
                out.write(flat.numpy().tobytes())
            print(f"    float x_mean[{X_mean.shape[0]}];")
            out.write(X_mean.numpy().tobytes())
            print(f"    float x_std[{X_std.shape[0]}];")
            out.write(X_std.numpy().tobytes())
            print(f"    float x_w[{X_w.shape[0]}];")
            out.write(X_w.numpy().tobytes())
            print(f"    float y_mean[{Y_mean.shape[0]}];")
            out.write(Y_mean.numpy().tobytes())
            print(f"    float y_std[{Y_std.shape[0]}];")
            out.write(Y_std.numpy().tobytes())
            print(f"    float x_idle[{x_lens.num_cols}];")
            out.write(X[0].numpy().tobytes())
            print(f"    float y_idle[{y_lens.num_cols}];")
            out.write(Y[0].numpy().tobytes())
            print(f"    float y_actual[{y_lens.num_cols}];")

            # evaluate the model on the X[0] and output the result.
            # this is used as a unit test on the c++ side.
            x_idle = X[0].unsqueeze(0)
            y_actual = model(x_idle, torch.tensor([0.0]))[0]
            assert y_actual.shape == (out_features,), y_actual.shape
            out.write(y_actual.numpy().tobytes())

            print(f"    int32_t parents[NUM_JOINTS];")
            parents = [skeleton.get_parent_index(skeleton.get_joint_name(i)) for i in range(skeleton.num_joints)]
            out.write(np.array(parents, dtype=np.int32).tobytes())
            print("};")

    print("const std::vector<Bone> pfnnBoneVec =")
    print("{")
    for i, name in enumerate(skeleton.joint_names):
        offset = skeleton.get_joint_offset(name)
        parent_index = skeleton.get_parent_index(name)
        print(f'    {{"{name}", {parent_index}, glm::mat4(1.0f, 0.0f, 0.0f, {offset[0]}f, 0.0f, 1.0f, 0.0f, {offset[1]}, 0.0f, 0.0f, 1.0f, {offset[2]}, 0.0f, 0.0f, 0.0f, 1.0f)}},')
    print("};")
