import glm
import mocap
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


def quat_swizzle(quat):
    return [quat[1], quat[2], quat[3], quat[0]]


def orient_towards(dir):
    return quat_swizzle(glm.quat(glm.vec3(1, 0, 0), dir))


def orient_line_from_pv(line, pos, vel):
    line.local.position = pos
    speed = glm.length(vel)
    if speed > 1e-6:
        line.local.rotation = orient_towards(vel / speed)
    else:
        line.local.rotation = quat_swizzle(glm.quat())
    SCALE_FACTOR = 0.1
    line.local.scale = max(speed * SCALE_FACTOR, 0.01)


class RenderBuddy:
    def __init__(
        self, skeleton, xforms, root, jointpva, traj, contacts, phase, rootvel
    ):
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

        self.draw_phase = False
        self.draw_root = True
        self.draw_joints = True
        self.draw_jointvel = False
        self.draw_traj = True
        self.draw_rootvel = False

        # use a group to position all elements that are in root-relative space
        self.root_group = gfx.Group()
        self.root_group.local.position = glm.vec3(self.root[0][3])
        self.root_group.local.rotation = quat_swizzle(glm.quat(self.root[0]))
        self.scene.add(self.root_group)

        if self.draw_root:
            # add a disc (flattened sphere) to display the root motion
            self.root_sphere = gfx.Mesh(
                gfx.sphere_geometry(1), gfx.MeshPhongMaterial(color="#aaaaff")
            )
            self.root_sphere.local.scale = glm.vec3(2, 0.01, 2)
            self.root_group.add(self.root_sphere)

        if self.draw_joints:
            joint_colors = {name: "#ffffff" for name in self.skeleton.joint_names}
            self.joint_mesh = []
            for i in range(skeleton.num_joints):
                joint_name = skeleton.get_joint_name(i)
                radius = 0.5

                # add a box to render each joint
                mesh = gfx.Mesh(
                    gfx.box_geometry(radius, radius, radius),
                    gfx.MeshPhongMaterial(color=joint_colors[joint_name]),
                )
                mesh.local.position = self.jointpva[0][i][0:3]
                mesh.local.rotation = quat_swizzle(
                    mocap.expmap(glm.vec3(self.jointpva[0][i][6:9]))
                )
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
                axes.local.position = glm.vec3(traj[0][o], 0, traj[0][o + 1])
                axes.local.rotation = orient_towards(
                    glm.vec3(traj[0][o + 2], 0, traj[0][o + 3])
                )
                self.traj_axes.append(axes)
                self.root_group.add(axes)

        self.camera = gfx.PerspectiveCamera(70, 4 / 3)
        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        # add a line for rendering phase as a clock
        if self.draw_phase:
            self.clock_group = gfx.Group()
            clock_hand = gfx.Mesh(
                gfx.box_geometry(0.1, 1, 0.1), gfx.MeshPhongMaterial(color="#ffffff")
            )
            clock_hand.local.position = [0, 0.5, 0]
            clock_dial = gfx.Mesh(
                gfx.sphere_geometry(1), gfx.MeshPhongMaterial(color="#0000ff")
            )
            clock_dial.local.scale = glm.vec3(1, 1, 0.001)
            self.clock_group.add(clock_hand)
            self.clock_group.add(clock_dial)
            self.scene.add(self.clock_group)

        if self.draw_rootvel:
            positions = [glm.vec3(r[3]) for r in self.root]
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
                orient_line_from_pv(
                    l, positions[i], FUDGE * glm.vec3(0, self.rootvel[i][2], 0)
                )
                self.scene.add(l)

        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.controller = gfx.OrbitController(
            camera=self.camera, register_events=self.renderer
        )

        self.renderer.add_event_handler(
            lambda event: self.on_key_down(event), "key_down"
        )

        self.canvas.request_draw(lambda: self.animate())

    def animate(self):
        if self.playing:
            self.curr_frame = self.curr_frame + 1
            if self.curr_frame >= self.end_frame or self.curr_frame >= len(self.xforms):
                self.curr_frame = self.start_frame

        # update root_group
        root_pos = glm.vec3(self.root[self.curr_frame][3])
        root_rot = quat_swizzle(glm.quat(self.root[self.curr_frame]))
        self.root_group.local.position = root_pos
        self.root_group.local.rotation = root_rot

        if self.draw_joints:
            for i in range(self.skeleton.num_joints):
                pos = glm.vec3(self.jointpva[self.curr_frame][i][0:3])
                rot = quat_swizzle(
                    mocap.expmap(glm.vec3(self.jointpva[self.curr_frame][i][6:9]))
                )
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
                pos = glm.vec3(self.jointpva[self.curr_frame][i][0:3])
                rot = quat_swizzle(
                    mocap.expmap(glm.vec3(self.jointpva[self.curr_frame][i][6:9]))
                )
                vel = glm.vec3(self.jointpva[self.curr_frame][i][3:6])
                orient_line_from_pv(self.joint_vels[i], pos, vel)

        # update trajectory
        if self.draw_traj:
            for i in range(TRAJ_WINDOW_SIZE):
                axes = self.traj_axes[i]
                o = i * TRAJ_ELEMENT_SIZE
                axes.local.position = glm.vec3(
                    traj[self.curr_frame][o], 0, traj[self.curr_frame][o + 1]
                )
                axes.local.rotation = orient_towards(
                    glm.vec3(
                        traj[self.curr_frame][o + 2], 0, traj[self.curr_frame][o + 3]
                    )
                )

        # animate phase
        if self.draw_phase:
            cam_pos = self.camera.world.position
            q = self.camera.world.rotation
            cam_rot = glm.quat(q[3], q[0], q[1], q[2])
            cam_xform = glm.mat4(cam_rot)
            cam_xform[3] = glm.vec4(glm.vec3(cam_pos), 1)
            offset_pos = glm.vec3(10, 7, -20)
            phase_spin = glm.angleAxis(-phase[self.curr_frame], glm.vec3(0, 0, 1))

            self.clock_group.world.position = cam_xform * offset_pos
            self.clock_group.world.rotation = quat_swizzle(cam_rot * phase_spin)

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

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle/load data
    skeleton = mocap.unpickle_obj(outbasepath + "_skeleton.pkl")
    xforms = mocap.unpickle_obj(outbasepath + "_xforms.pkl")
    root = mocap.unpickle_obj(outbasepath + "_root.pkl")
    jointpva = np.load(outbasepath + "_jointpva.npy")
    traj = np.load(outbasepath + "_traj.npy")
    contacts = np.load(outbasepath + "_contacts.npy")
    phase = np.load(outbasepath + "_phase.npy")
    rootvel = np.load(outbasepath + "_rootvel.npy")

    renderBuddy = RenderBuddy(
        skeleton, xforms, root, jointpva, traj, contacts, phase, rootvel
    )
    run()
