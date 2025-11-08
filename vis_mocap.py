#
# Copyright (c) 2025 Anthony J. Thibault
# This software is licensed under the MIT License. See LICENSE for more details.
#

import math_util as mu
from skeleton import Skeleton
import numpy as np
import os
import pickle
import pygfx as gfx
from tqdm import trange, tqdm
import sys
from wgpu.gui.auto import WgpuCanvas, run


OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def orient_towards(dir):
    return mu.quat_from_vectors(np.array([1, 0, 0]), dir)


def orient_line_from_pv(line, pos, vel):
    line.local.position = pos
    speed = np.linalg.norm(vel)
    if speed > 1e-6:
        line.local.rotation = orient_towards(vel / speed)
    else:
        line.local.rotation = np.array([0, 0, 0, 1])
    SCALE_FACTOR = 0.1
    line.local.scale = max(speed * SCALE_FACTOR, 0.01)


class RenderBuddy:
    def __init__(self, skeleton, xforms, root, jointpva, traj, contacts, phase, rootvel):
        self.start_frame = 0
        self.end_frame = sys.maxsize
        self.curr_frame = self.start_frame
        self.playing = True

        self.skeleton = skeleton
        self.xforms = xforms
        self.root = root
        self.jointpva = jointpva
        self.traj = traj
        self.contacts = contacts
        self.phase = phase
        self.rootvel = rootvel

        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight(intensity=1))
        self.scene.add(gfx.DirectionalLight())
        self.scene.add(gfx.helpers.AxesHelper(10.0, 0.5))
        self.scene.add(gfx.helpers.GridHelper(size=100))

        self.draw_phase = True
        self.draw_root = False
        self.draw_joints = True
        self.draw_jointvel = False
        self.draw_traj = False
        self.draw_rootvel = False

        # use a group to position all elements that are in root-relative space
        self.root_group = gfx.Group()
        self.root_group.local.position = self.root[0, 0:3, 3]
        self.root_group.local.rotation = mu.quat_from_mat(self.root[0])
        self.scene.add(self.root_group)

        if self.draw_root:
            # add a disc (flattened sphere) to display the root motion
            self.root_sphere = gfx.Mesh(gfx.sphere_geometry(1), gfx.MeshPhongMaterial(color="#aaaaff"))
            self.root_sphere.local.scale = [2, 0.01, 2]
            self.root_group.add(self.root_sphere)

        if self.draw_joints:
            joint_colors = {name: "#ffffff" for name in self.skeleton.joint_names}
            self.joint_mesh = []
            for i in range(skeleton.num_joints):
                joint_name = skeleton.get_joint_name(i)
                radius = 0.5

                # add a box to render each joint
                """
                mesh = gfx.Mesh(
                    gfx.box_geometry(radius, radius, radius),
                    gfx.MeshPhongMaterial(color=joint_colors[joint_name]),
                )
                """
                mesh = gfx.helpers.AxesHelper(radius * 2, 1)
                mesh.local.position = self.jointpva[0][i][0:3]

                rot6d = self.jointpva[0][i][6:12]
                # convert into a quaternion
                mat = np.eye(3)
                x_axis = rot6d[0:3]
                y_axis = rot6d[3:6]
                z_axis = np.linalg.cross(x_axis, y_axis)
                y_axis = np.linalg.cross(z_axis, x_axis)
                mat[0:3, 0] = x_axis
                mat[0:3, 1] = y_axis
                mat[0:3, 2] = z_axis
                mesh.local.rotation = mu.quat_from_mat(mat)

                self.joint_mesh.append(mesh)
                self.root_group.add(mesh)

        if self.draw_jointvel:
            self.joint_vels = []
            for i in range(skeleton.num_joints):
                # add line to render joint velocity
                line = gfx.Line(
                    gfx.Geometry(positions=[[0, 0, 0], [1, 0, 0]]),
                    gfx.LineMaterial(thickness=2.0, color="#ff0000"),
                )
                line.local.position = self.jointpva[0][i][0:3]
                line.local.scale = 0.1
                self.joint_vels.append(line)
                self.root_group.add(line)

        # add an axes helper for each element of the trajectory
        if self.draw_traj:
            self.traj_axes = []
            for i in range(TRAJ_WINDOW_SIZE):
                axes = gfx.helpers.AxesHelper(3.0, 0.5)
                o = i * TRAJ_ELEMENT_SIZE
                axes.local.position = [traj[0][o], 0, traj[0][o + 1]]
                axes.local.rotation = orient_towards(np.array([traj[0][o + 2], 0, traj[0][o + 3]]))
                self.traj_axes.append(axes)
                self.root_group.add(axes)

        self.camera = gfx.PerspectiveCamera(70, 4 / 3)
        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        # add a line for rendering phase as a clock
        if self.draw_phase:
            self.clock_group = gfx.Group()
            clock_hand = gfx.Mesh(gfx.box_geometry(0.1, 1, 0.1), gfx.MeshPhongMaterial(color="#ffffff"))
            clock_hand.local.position = [0, 0.5, 0]
            clock_dial = gfx.Mesh(gfx.sphere_geometry(1), gfx.MeshPhongMaterial(color="#0000ff"))
            clock_dial.local.scale = [1, 1, 0.001]
            self.clock_group.add(clock_hand)
            self.clock_group.add(clock_dial)
            self.scene.add(self.clock_group)

        if self.draw_rootvel:
            positions = [r[0:3, 3] for r in self.root]
            root_line = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineMaterial(thickness=2.0, color="#0000ff"),
            )
            self.scene.add(root_line)

            for i in range(len(self.root)):
                l = gfx.Line(
                    gfx.Geometry(positions=[[0, 0, 0], [1, 0, 0]]),
                    gfx.LineMaterial(thickness=1.0, color="#00ff00"),
                )
                FUDGE = 20  # make the angular vels longer to visualize them better
                orient_line_from_pv(l, positions[i], FUDGE * np.array([0, self.rootvel[i][2], 0]))
                self.scene.add(l)

        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.controller = gfx.OrbitController(camera=self.camera, register_events=self.renderer)

        self.renderer.add_event_handler(lambda event: self.on_key_down(event), "key_down")

        self.canvas.request_draw(lambda: self.animate())

    def animate(self):
        if self.playing:
            self.curr_frame = self.curr_frame + 1
            if self.curr_frame >= self.end_frame or self.curr_frame >= xforms.shape[0]:
                self.curr_frame = self.start_frame

        # update root_group
        root_pos = self.root[self.curr_frame, 0:3, 3]
        root_rot = mu.quat_from_mat(self.root[self.curr_frame])
        self.root_group.local.position = root_pos
        self.root_group.local.rotation = root_rot

        if self.draw_joints:
            for i in range(self.skeleton.num_joints):
                pos = self.jointpva[self.curr_frame][i][0:3]
                rot = mu.expmap(self.jointpva[self.curr_frame][i][6:9])
                self.joint_mesh[i].local.position = pos
                self.joint_mesh[i].local.rotation = rot

            # update contacts
            feet = [
                self.skeleton.get_joint_index("LeftFoot"),
                self.skeleton.get_joint_index("RightFoot"),
                self.skeleton.get_joint_index("LeftToeBase"),
                self.skeleton.get_joint_index("RightToeBase"),
            ]
            for i, joint in enumerate(feet):
                if self.contacts[self.curr_frame][i] > 0.5:
                    self.joint_mesh[joint].material.color = "#ff0000"
                    self.joint_mesh[joint].local.scale = 2
                else:
                    self.joint_mesh[joint].material.color = "#00ffff"
                    self.joint_mesh[joint].local.scale = 1

        if self.draw_jointvel:
            for i in range(self.skeleton.num_joints):
                pos = self.jointpva[self.curr_frame][i][0:3]
                rot = mu.expmap(self.jointpva[self.curr_frame][i][6:9])
                vel = self.jointpva[self.curr_frame][i][3:6]
                orient_line_from_pv(self.joint_vels[i], pos, vel)

        # update trajectory
        if self.draw_traj:
            for i in range(TRAJ_WINDOW_SIZE):
                axes = self.traj_axes[i]
                o = i * TRAJ_ELEMENT_SIZE
                axes.local.position = [
                    traj[self.curr_frame][o],
                    0,
                    traj[self.curr_frame][o + 1],
                ]
                axes.local.rotation = orient_towards(
                    np.array([traj[self.curr_frame][o + 2], 0, traj[self.curr_frame][o + 3]])
                )

        # animate phase
        if self.draw_phase:
            cam_xform = mu.build_mat_from_quat(np.eye(4), np.array(self.camera.world.rotation))
            cam_xform[0:3, 3] = self.camera.world.position
            offset_pos = np.array([10, 7, -20, 1])
            phase_xform = mu.build_mat_rotz(np.eye(4), -self.phase[self.curr_frame])

            self.clock_group.world.position = (cam_xform @ offset_pos)[0:3]
            self.clock_group.world.rotation = mu.quat_from_mat(cam_xform @ phase_xform)

        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

    def on_key_down(self, event):
        if event.key == "Escape":
            self.renderer.target.close()
        elif event.key == " ":
            self.playing = not self.playing


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    anim = sys.argv[1]

    # unpickle/load data
    skeleton = unpickle_obj(os.path.join(OUTPUT_DIR, "skeleton", anim + ".pkl"))
    xforms = np.load(os.path.join(OUTPUT_DIR, "xforms", anim + ".npy"))
    root = np.load(os.path.join(OUTPUT_DIR, "root", anim + ".npy"))
    jointpva = np.load(os.path.join(OUTPUT_DIR, "jointpva", anim + ".npy"))
    traj = np.load(os.path.join(OUTPUT_DIR, "traj", anim + ".npy"))
    contacts = np.load(os.path.join(OUTPUT_DIR, "contacts", anim + ".npy"))
    phase = np.load(os.path.join(OUTPUT_DIR, "phase", anim + ".npy"))
    rootvel = np.load(os.path.join(OUTPUT_DIR, "rootvel", anim + ".npy"))

    renderBuddy = RenderBuddy(skeleton, xforms, root, jointpva, traj, contacts, phase, rootvel)
    run()
