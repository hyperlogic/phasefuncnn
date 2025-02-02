Implementation of PFNN
-------------------------
build_xforms.py - generates skeleton, world space xforms and world space root motion
build_jointpva.py - generates root relative joint positions, velocities and angles (logmap)
build_traj.py - generates 2d trajectory samples for each frame.
build_contacts.py - generates contact info and gait phase
build_tensors.py - generates pytorch tensors for X, Y and P for each anim
normalize_tensors.py - concatinates all X, Y, and Ps and normalizes data for training.

See datalens.py for a helper class used to help accesing columns of input/output tensors by name rather then number.

See Snakefile for rules which invoke each step in the data preperation process.

intermediate data files
-------------------------
skeleton/*.pkl - pickled mocap.Skeleton instance
xforms/*.npy - world space transforms of each joint for each frame.
    numpy array with shape (num_frames, num_joints, 4, 4)
root/*.npy - world space transforms for the character root motion.
    with with shape (num_frames, 4, 4)
jointpva/*.npy - np.ndarray of shape (num_frames, num_joints, 9) where each element is
    an array with 9 elements:
    [0:3] (px, py, py) is the position of the joint in root-relative space
    [3:6] (vx, vy, vz) the joint velocity in root-relative space
    [6:9] (ax, ay, az) the joint orientation in R^3 use mocap.expmap to map to a quaternion.
traj/*.npy - np.ndarray of shape (num_frames, TRAJ_WINDOW_SIZE * TRAJ_ELEMENT_SIZE)
    TRAJ_WINDOW_SIZE is the number of trajectory samples around the current frame.
    TRAJ_ELEMENT_SIZE is 4, [0:2] (px, pz) position of trajectory sample (relative to root of current frame)
    [2:4] (dx, dy) direction of motion along trajectory (relative to root of current frame)
rootvel/*.npy - np.ndarray of shape (num_frames, 2) [0:2] (vx, vz) velocity of motion in root frame
    [2:3] (angv) angular vel of motion in root frame
contacts/*.npy - np.ndarray of shape (num_frames, 4) [0:4] (lfoot, rfoot, ltoe, rtoe) - 1 if foot is in contact with ground.
phase/*.npy - np.ndarray of shape (num_frames) phase of each frame. [0, 2*pi)
gait/*.npy - np.ndarray of shape (num_frames, 8) one-hot vectors identify gait
    (stand, walk, jog, run, crouch, jump, crawl, unkown)
tensors/*_X.pth - output tensor for a single animation
tensors/*_X_std.pth, tensors/*_X_mean.pth, tensors/*_X_w.pth - torch tensors used for normaliztion during inference
tensors/*_Y.pth - output tensors for single anim
tensors/*_Y_std.pth, tensors/*_Y_mean.pth - used to un-normalize output during inference


final output
-----------------
X.pth - torch input tensor
X_mean.pth - mean of input tensor
X_std.pth - standard deviation of input tensor
X_w.pth - weights used to reduce importance of joint features
Y.pth - torch output tensor
Y_mean.pth - mean of output tensor
Y_std.pth - standard deviation of output tensor
P.pth - phase tensor


Input tensor (X.pth)
------------
x = { trajpd_i trajd_i jointp_i-1 jointv_i-1 gait_i }

trajpd_i - TRAJ_WINDOW_SIZE * 4 floats total
jointpv_i-1 - NUM_JOINTS * 6 floats total
gait_i - 8 floats

shape = (num_rows, TRAJ_WINDOW_SIZE * 4 + num_joints * 6 + 8)

x is then normalized and weighted
  * except gait is not normalized (by setting mean=0 std=1)
  * only joint pos and vel are weighted, by 0.1 currently. gait and traj have weight of 1.


Output tensor (Y.pth)
-------------------
y = { trajp_i+1 trajd_i+1 jointp_i jointv_i jointa_i rootvel_i, phasevel_i, contacts_i }

trajpd_i+1 - TRAJ_WINDOW_SIZE * 4 floats total
jointpva_i - NUM_JOINTS * 9 floats total
rootvel_i - 3 floats (vx, vy, angvel)
phasevel_i - 1 float
contacts_i - 4 floats

shape = (num_rows, TRAJ_WINDOW_SIZE * 4 + num_joints * 9 + 9)

y is then normalized
  * TODO: should probably not weight one-hot contact vectors.


Phase tensor (P.pth)
-----------------
phase_i-1 - 1 float

shape = (num_rows, 1)



TODO:
--------------
* currently training hits a brick wall very quickly, why?
  * YES: lambda too high?

* NO: Should gait be normalized or not? NO it should not

* Should I switch from l1 regularisation of model weights to AdamW with weight decay?
* Should I alter joint importance scaling?
* Is my learning rate too high?  to low?

* training should save a readable name and or other parameters per experiment.
* training should be able to start from a previous checkpoint.

* There is some weirdness in input data.
  * right foot tweaks out to the side sometimes.
  * right foot goes pigpen toe sometimes
  * right arm doesn't swing as much as left arm.

* Should I add rest.bvh to training set?

Gait questions
------------------
* how much gait coverage do I have? in training set?
* how should I pick gait during inference?
* What speed matches each gait the best?

* Add jump!
* Add leg IK

* Speed up inference.

* allow pfnn to work with non batched x, and phase, for convenience
  and speed during inference.  i.e. dont have to do catmul-rom 4 times

* draw contacts
* TODO: may have to keep contacts un-normalized in training data.

* Better fix follow cam
