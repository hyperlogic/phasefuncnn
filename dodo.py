from doit.action import CmdAction
import importlib
import os
import pkgutil


OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 6  # 6 for px, py, pz, dx, dy, dz

mocap_paths = ["../PFNN/data/animations/LocomotionFlat09_000.bvh"]
# mocap_paths = ["../PFNN/data/animations/LocomotionFlat01_000.bvh"]
mocap = [os.path.splitext(os.path.basename(n))[0] for n in mocap_paths]
xform_targets = (
    [os.path.join(OUTPUT_DIR, m + "_xforms.pkl") for m in mocap]   # AJT: TODO REMOVE
    + [os.path.join(OUTPUT_DIR, m + "_skeleton.pkl") for m in mocap]
    + [os.path.join(OUTPUT_DIR, m + "_root.pkl") for m in mocap]
    + [os.path.join(OUTPUT_DIR, m + "_xforms.npy") for m in mocap]
)
jointpva_targets = [os.path.join(OUTPUT_DIR, m + "_jointpva.npy") for m in mocap]
traj_targets = [os.path.join(OUTPUT_DIR, m + "_traj.npy") for m in mocap] + [
    os.path.join(OUTPUT_DIR, m + "_rootvel.npy") for m in mocap
]
contacts_targets = [os.path.join(OUTPUT_DIR, m + "_contacts.npy") for m in mocap] + [
    os.path.join(OUTPUT_DIR, m + "_phase.npy") for m in mocap
]


def get_python_files_in_module(module_name):
    result = []
    module = importlib.import_module(module_name)
    for importer, mod_name, is_pkg in pkgutil.walk_packages(module.__path__, module_name + "."):
        if not is_pkg:
            result.append(mod_name.replace(".", "/") + ".py")
    return result

mocap_deps = get_python_files_in_module("mocap")


def task_build_xforms():
    """Build world space transform matrices for each joint and root motion, from a bvh file"""
    code_deps = [__file__, "build_xforms.py"] + mocap_deps
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
    code_deps = [__file__, "build_jointpva.py"] + mocap_deps
    return {
        "file_dep": code_deps + xform_targets,
        "targets": jointpva_targets,
        "actions": [f"python build_jointpva.py {m}" for m in mocap],
        "clean": True,
    }


def task_build_traj():
    """Build root-space trajectory window around each frame."""
    code_deps = [__file__, "build_traj.py"] + mocap_deps
    return {
        "file_dep": code_deps + jointpva_targets,
        "targets": traj_targets,
        "actions": [f"python build_traj.py {m}" for m in mocap],
        "clean": True,
    }


def task_build_contacts():
    """Build root-space trajectory window around each frame."""
    code_deps = [__file__, "build_contacts.py"] + mocap_deps
    return {
        "file_dep": code_deps + xform_targets,
        "targets": contacts_targets,
        "actions": [f"python build_contacts.py {m}" for m in mocap],
        "clean": True,
    }
