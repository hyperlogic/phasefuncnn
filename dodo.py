from doit.action import CmdAction
import os


OUTPUT_DIR = "output"

mocap_paths = ["../PFNN/data/animations/LocomotionFlat09_000.bvh"]
mocap = [os.path.splitext(os.path.basename(n))[0] for n in mocap_paths]
xform_targets = [os.path.join(OUTPUT_DIR, m + "_xforms.pkl") for m in mocap] + [
    os.path.join(OUTPUT_DIR, m + "_skeleton.pkl") for m in mocap
]
vel_targets = [os.path.join(OUTPUT_DIR, m + "_vels.pkl") for m in mocap]
jposdir_targets = [os.path.join(OUTPUT_DIR, m + "_jposdir.pkl") for m in mocap]


def task_build_xforms():
    """Convert bvh files to xforms"""
    code_deps = [__file__, "build_xforms.py"]
    return {
        "file_dep": code_deps + mocap_paths,
        "targets": xform_targets,
        "actions": [
            CmdAction(f"python build_xforms.py {p}", buffering=1) for p in mocap_paths
        ],
    }


def task_build_vels():
    """Convert xforms to velocities"""
    code_deps = [__file__, "build_vels.py"]
    return {
        "file_dep": code_deps + xform_targets,
        "targets": vel_targets,
        "actions": [f"python build_vels.py {m}" for m in mocap],
    }


def task_build_jposdir():
    """Convert xforms to joint positions and dirs relative to root motion (jposdir)"""
    code_deps = [__file__, "build_jposdir.py"]
    return {
        "file_dep": code_deps + vel_targets,
        "targets": jposdir_targets,
        "actions": [f"python build_jposdir.py {m}" for m in mocap],
    }
