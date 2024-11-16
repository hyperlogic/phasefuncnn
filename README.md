Implementation of PFNN
-------------------------
build_xforms.py - generates skeleton, world space xforms and world space root motion
build_jointpva.py - generates root relative joint positions, velocities and angles (logmap)
build_traj.py - generates 2d trajectory samples for each frame.
build_contacts.py - generates contact info and gait phase
build_tensors.py - generates pytorch tensors for X, Y and P ready for training

intermediate data files
-------------------------
*_skeleton.pkl - pickled mocap.Skeleton instance
*_xforms.npy - world space transforms of each joint for each frame.
    numpy array with shape (num_frames, num_joints, 4, 4)
*_root.npy - world space transforms for the character root motion.
    with with shape (num_frames, 4, 4)
*_jointpva.npy - np.ndarray of shape (num_frames, num_joints, 9) where each element is
    an array with 9 elements:
    [0:3] (px, py, py) is the position of the joint in root-relative space
    [3:6] (vx, vy, vz) the joint velocity in root-relative space
    [6:9] (ax, ay, az) the joint orientation in R^3 use mocap.expmap to map to a quaternion.
*_traj.npy - np.ndarray of shape (num_frames, TRAJ_WINDOW_SIZE * TRAJ_ELEMENT_SIZE)
    TRAJ_WINDOW_SIZE is the number of trajectory samples around the current frame.
    TRAJ_ELEMENT_SIZE is 4, [0:2] (px, pz) position of trajectory sample (relative to root of current frame)
    [2:4] (dx, dy) direction of motion along trajectory (relative to root of current frame)
*_rootvel.npy - np.ndarray of shape (num_frames, 2) [0:2] (vx, vz) velocity of motion in root frame
    [2:3] (angv) angular vel of motion in root frame
*_contacts.npy - np.ndarray of shape (num_frames, 4) [0:4] (lfoot, rfoot, ltoe, rtoe) - 1 if foot is in contact with ground.
*_phase.npy - np.ndarray of shape (num_frames) phase of each frame. [0, 2*pi)


Input tensor
------------
x = { trajpd_i trajd_i jointp_i-1 jointv_i-1f }

trajpd_i - TRAJ_WINDOW_SIZE * 4 floats total
jointpv_i-1 - NUM_JOINTS * 6 floats total

shape = (num_rows, TRAJ_WINDOW_SIZE * 4 + num_joints * 6)


Output tensor
-------------------
y = { trajp_i+1 trajd_i+1 jointp_i jointv_i jointa_i rootvel_i, phasevel_i, contacts_i }

trajpd_i+1 - TRAJ_WINDOW_SIZE * 4 floats total
jointpva_i - NUM_JOINTS * 9 floats total
rootvel_i - 3 floats (vx, vy, angvel)
phasevel_i - 1 float
contacts_i - 4 floats

shape = (num_rows, TRAJ_WINDOW_SIZE * 4 + num_joints * 9 + 9)

Phase tensor
-----------------
phase_i-1 - 1 float

shape = (num_rows, 1)

TODO:
-------------------------
BUILD
-----------
* normalization of inputs -
  In build process, generate ALL data as a tensor.
  * normalize and importance scale ALL data during this process.
  * resulting in X, Y and P tensors
* make sure concatination of all mocap files works.

TRAIN
----------
* correct loss function
* phase function of cubic catmull rom
* dropout layers




