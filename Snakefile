import pathlib

PFNN_DIR = pathlib.Path("../PFNN/data/animations")
OUTPUT_DIR = pathlib.Path("output")

ANIMS = [
    "LocomotionFlat01_000",
    "LocomotionFlat02_000",
    "LocomotionFlat02_001",
    "LocomotionFlat03_000",
    "LocomotionFlat04_000",
    "LocomotionFlat05_000",
    "LocomotionFlat06_000",
    "LocomotionFlat06_001",
    "LocomotionFlat07_000",
    "LocomotionFlat08_000",
    "LocomotionFlat08_001",
    "LocomotionFlat09_000",
    "LocomotionFlat10_000",
    "LocomotionFlat11_000",
    "LocomotionFlat12_000",
    "NewCaptures01_000",
    "NewCaptures02_000",
    "NewCaptures03_000",
    "NewCaptures03_001",
    "NewCaptures03_002",
    "NewCaptures04_000",
    "NewCaptures05_000",
    "NewCaptures07_000",
    "NewCaptures08_000",
    "NewCaptures09_000",
    "NewCaptures10_000",
    "NewCaptures11_000",
]

ANIMS = ANIMS + [a + "_mirror" for a in ANIMS]


rule all:
    input:
        OUTPUT_DIR / "final_checkpoint.pth"


rule train:
    input:
        OUTPUT_DIR / "tensors/X.pth",
        OUTPUT_DIR / "tensors/X_mean.pth",
        OUTPUT_DIR / "tensors/X_std.pth",
        OUTPUT_DIR / "tensors/X_w.pth",
        OUTPUT_DIR / "tensors/Y.pth",
        OUTPUT_DIR / "tensors/Y_mean.pth",
        OUTPUT_DIR / "tensors/Y_std.pth",
        OUTPUT_DIR / "tensors/P.pth",
    output:
        OUTPUT_DIR / "final_checkpoint.pth"
    shell:
        "python train.py"


rule build_xforms:
    input:
        bvh=PFNN_DIR / "{anim}.bvh"
    output:
        skeleton=OUTPUT_DIR / "skeleton/{anim}.pkl",
        xforms=OUTPUT_DIR / "xforms/{anim}.npy",
        root=OUTPUT_DIR / "root/{anim}.npy",
    params:
        mirror=False
    script:
        "build_xforms.py"


rule build_jointpva:
    input:
        skeleton=OUTPUT_DIR / "skeleton/{anim}.pkl",
        xforms=OUTPUT_DIR / "xforms/{anim}.npy",
        root=OUTPUT_DIR / "root/{anim}.npy",
    output:
        jointpva=OUTPUT_DIR / "jointpva/{anim}.npy"
    script:
        "build_jointpva.py"


rule build_traj:
     input:
        root=OUTPUT_DIR / "root/{anim}.npy"
     output:
        traj=OUTPUT_DIR / "traj/{anim}.npy",
        rootvel=OUTPUT_DIR / "rootvel/{anim}.npy"
     script:
        "build_traj.py"


rule build_contacts:
    input:
        skeleton=OUTPUT_DIR / "skeleton/{anim}.pkl",
        xforms=OUTPUT_DIR / "xforms/{anim}.npy",
        phase=PFNN_DIR / "{anim}.phase",
        gait=PFNN_DIR / "{anim}.gait",
    output:
        contacts=OUTPUT_DIR / "contacts/{anim}.npy",
        phase=OUTPUT_DIR / "phase/{anim}.npy",
        gait=OUTPUT_DIR / "gait/{anim}.npy",
    script:
        "build_contacts.py"


rule build_tensors:
    input:
        skeleton_list=expand(OUTPUT_DIR / "skeleton/{anim}.pkl", anim=ANIMS),
        root_list=expand(OUTPUT_DIR / "root/{anim}.npy", anim=ANIMS),
        jointpva_list=expand(OUTPUT_DIR / "jointpva/{anim}.npy", anim=ANIMS),
        traj_list=expand(OUTPUT_DIR / "traj/{anim}.npy", anim=ANIMS),
        rootvel_list=expand(OUTPUT_DIR / "rootvel/{anim}.npy", anim=ANIMS),
        contacts_list=expand(OUTPUT_DIR / "contacts/{anim}.npy", anim=ANIMS),
        phase_list=expand(OUTPUT_DIR / "phase/{anim}.npy", anim=ANIMS),
        gait_list=expand(OUTPUT_DIR / "gait/{anim}.npy", anim=ANIMS),
    output:
        x=OUTPUT_DIR / "tensors/X.pth",
        x_mean=OUTPUT_DIR / "tensors/X_mean.pth",
        x_std=OUTPUT_DIR / "tensors/X_std.pth",
        x_w=OUTPUT_DIR / "tensors/X_w.pth",
        y=OUTPUT_DIR / "tensors/Y.pth",
        y_mean=OUTPUT_DIR / "tensors/Y_mean.pth",
        y_std=OUTPUT_DIR / "tensors/Y_std.pth",
        p=OUTPUT_DIR / "tensors/P.pth",
    script:
        "build_tensors.py"