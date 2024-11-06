from doit.action import CmdAction
import os


OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 6  # 6 for px, py, pz, dx, dy, dz

mocap_paths = ["../PFNN/data/animations/LocomotionFlat09_000.bvh"]
# mocap_paths = ["../PFNN/data/animations/LocomotionFlat01_000.bvh"]
mocap = [os.path.splitext(os.path.basename(n))[0] for n in mocap_paths]
xform_targets = (
    [os.path.join(OUTPUT_DIR, m + "_xforms.pkl") for m in mocap]
    + [os.path.join(OUTPUT_DIR, m + "_skeleton.pkl") for m in mocap]
    + [os.path.join(OUTPUT_DIR, m + "_root.pkl") for m in mocap]
)
jointpva_targets = [os.path.join(OUTPUT_DIR, m + "_jointpva.npy") for m in mocap]
traj_targets = [os.path.join(OUTPUT_DIR, m + "_traj.npy") for m in mocap]


def task_build_xforms():
    """Build world space transform matrices for each joint and root motion, from a bvh file"""
    code_deps = [__file__, "build_xforms.py"]
    return {
        "file_dep": code_deps + mocap_paths,
        "targets": xform_targets,
        "actions": [
            CmdAction(f"python build_xforms.py {p}", buffering=1) for p in mocap_paths
        ],
        "clean": True,
    }


def task_build_jointpva():
    """Build root-space position, velocity and angles (pva) for each joint"""
    code_deps = [__file__, "build_jointpva.py"]
    return {
        "file_dep": code_deps + xform_targets,
        "targets": jointpva_targets,
        "actions": [f"python build_jointpva.py {m}" for m in mocap],
        "clean": True,
    }


def task_build_traj():
    """Build root-space trajectory window around each frame."""
    code_deps = [__file__, "build_traj.py"]
    return {
        "file_dep": code_deps + jointpva_targets,
        "targets": traj_targets,
        "actions": [f"python build_traj.py {m}" for m in mocap],
        "clean": True,
    }
