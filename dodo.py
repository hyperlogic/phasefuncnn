from doit.action import CmdAction
import os


OUTPUT_DIR = "output"

mocap_paths = ["../PFNN/data/animations/LocomotionFlat09_000.bvh"]
mocap = [os.path.splitext(os.path.basename(n))[0] for n in mocap_paths]


def task_build_xforms():
    """Convert bvh files to xforms"""
    return {
        "file_dep": ["bvh_to_xforms.py", "mocap/build_xforms.py"] + mocap_paths,
        "targets": (
            [os.path.join(OUTPUT_DIR, m + "_xforms.pkl") for m in mocap]
            + [os.path.join(OUTPUT_DIR, m + "_skeleton.pkl") for m in mocap]
        ),
        "actions": [CmdAction(f"python bvh_to_xforms.py {p}", buffering=1) for p in mocap_paths],
    }


def task_build_vels():
    """Convert xforms to velocities"""
    return {
        "file_dep": (
            [os.path.join(OUTPUT_DIR, m + "_xforms.pkl") for m in mocap]
            + [os.path.join(OUTPUT_DIR, m + "_skeleton.pkl") for m in mocap]
        ),
        "targets": [os.path.join(OUTPUT_DIR, m + "_vels.pkl") for m in mocap],
        "actions": [f"python xforms_to_vels.py {m}" for m in mocap],
    }
