from doit.action import CmdAction
import importlib
import os
import pathlib
import pkgutil


OUTPUT_DIR = pathlib.Path("output")

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

mocap_paths = ["../PFNN/data/animations/LocomotionFlat09_000.bvh"]

mocap_names = [os.path.splitext(os.path.basename(n))[0] for n in mocap_paths]

def out_deps(basename, filenames):
    return [OUTPUT_DIR / f"{basename}_{filename}" for filename in filenames]


def get_python_files_in_module(module_name):
    result = []
    module = importlib.import_module(module_name)
    for importer, mod_name, is_pkg in pkgutil.walk_packages(
        module.__path__, module_name + "."
    ):
        if not is_pkg:
            result.append(mod_name.replace(".", "/") + ".py")
    return result


mocap_deps = get_python_files_in_module("mocap")


def task_build_xforms():
    """Build world space transform matrices for each joint and root motion, from a bvh file"""
    code_deps = [__file__, "build_xforms.py"] + mocap_deps
    for i, name in enumerate(mocap_names):
        yield {
            "name": name,
            "file_dep": code_deps + [mocap_paths[i]],
            "targets": out_deps(name, ["skeleton.pkl", "xforms.npy", "root.npy"]),
            "actions": [f"python build_xforms.py {mocap_paths[i]}"],
            "clean": True,
        }


def task_build_jointpva():
    """Build root-space position, velocity and angles (pva) for each joint"""
    code_deps = [__file__, "build_jointpva.py"] + mocap_deps
    for name in mocap_names:
        yield {
            "name": name,
            "file_dep": code_deps
            + out_deps(name, ["skeleton.pkl", "xforms.npy", "root.npy"]),
            "targets": out_deps(name, ["jointpva.npy"]),
            "actions": [f"python build_jointpva.py {name}"],
            "clean": True,
        }


def task_build_traj():
    """Build root-space trajectory window around each frame."""
    code_deps = [__file__, "build_traj.py"] + mocap_deps
    for name in mocap_names:
        yield {
            "name": name,
            "file_dep": code_deps + out_deps(name, ["root.npy"]),
            "targets": out_deps(name, ["traj.npy", "rootvel.npy"]),
            "actions": [f"python build_traj.py {name}"],
            "clean": True,
        }


def task_build_contacts():
    """Build root-space trajectory window around each frame."""
    code_deps = [__file__, "build_contacts.py"] + mocap_deps
    for name in mocap_names:
        yield {
            "name": name,
            "file_dep": code_deps + out_deps(name, ["skeleton.pkl", "xforms.npy"]),
            "targets": out_deps(name, ["contacts.npy", "phase.npy"]),
            "actions": [f"python build_contacts.py {name}"],
            "clean": True,
        }


def task_build_tensors():
    """Build fully normalized pytorch tensors for X, Y, P"""
    file_deps = [__file__, "build_tensors.py"] + mocap_deps
    for name in mocap_names:
        file_deps += out_deps(
            name,
            [
                "skeleton.pkl",
                "root.npy",
                "jointpva.npy",
                "traj.npy",
                "contacts.npy",
                "phase.npy",
                "rootvel.npy",
            ],
        )
    return {
        "file_dep": file_deps,
        "targets": [
            OUTPUT_DIR / "X.pth",
            OUTPUT_DIR / "X_mean.pth",
            OUTPUT_DIR / "X_std.pth",
            OUTPUT_DIR / "X_w.pth",
            OUTPUT_DIR / "Y.pth",
            OUTPUT_DIR / "Y_mean.pth",
            OUTPUT_DIR / "Y_std.pth",
            OUTPUT_DIR / "P.pth",
        ],
        "actions": [
            CmdAction(f"python build_tensors.py {' '.join(mocap_names)}", buffering=1)
        ],
        "clean": True,
    }
