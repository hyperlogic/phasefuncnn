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


def quat_swizzle(quat):
    return [quat[1], quat[2], quat[3], quat[0]]


class RenderBuddy:
    def __init__(self, skeleton, xforms, root, jointpva):

        self.start_frame = 0
        self.end_frame = sys.maxsize
        self.curr_frame = self.start_frame
        self.playing = True

        self.skeleton = skeleton
        self.xforms = xforms
        self.root = root

        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight(intensity=1))
        self.scene.add(gfx.DirectionalLight())
        self.scene.add(gfx.helpers.AxesHelper(10.0, 0.5))
        self.scene.add(gfx.helpers.GridHelper(size=100))

        # draw a disc under the root position.
        self.root_group = gfx.helpers.AxesHelper(10.0, 1.0)  # gfx.Group()
        self.root_group.local.position = glm.vec3(self.root[0][3])
        self.root_group.local.rotation = quat_swizzle(glm.quat(self.root[0]))
        self.scene.add(self.root_group)

        self.root_sphere = gfx.Mesh(
            gfx.sphere_geometry(1), gfx.MeshPhongMaterial(color="#aaaaff")
        )
        self.root_sphere.local.scale = glm.vec3(2, 0.01, 2)
        self.root_group.add(self.root_sphere)

        # build sphere for every transform
        joint_colors = {name: "#ffffff" for name in self.skeleton.joint_names}
        joint_colors["LeftUpLeg"] = "#0000ff"
        joint_colors["RightUpLeg"] = "#ff0000"
        joint_colors["LeftArm"] = "#0000ff"
        joint_colors["RightArm"] = "#ff0000"
        self.spheres = []
        for i in range(skeleton.num_joints):
            joint_name = skeleton.get_joint_name(i)
            radius = 0.5
            sphere = gfx.Mesh(
                gfx.box_geometry(radius, radius, radius),
                gfx.MeshPhongMaterial(color=joint_colors[joint_name]),
            )
            sphere.local.position = jointpva[0][i][0:3]
            sphere.local.rotation = quat_swizzle(
                mocap.expmap(glm.vec3(jointpva[0][i][6:9]))
            )
            self.spheres.append(sphere)
            self.root_group.add(sphere)

        self.camera = gfx.PerspectiveCamera(70, 4 / 3)
        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

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

        for i in range(self.skeleton.num_joints):
            self.spheres[i].local.position = jointpva[self.curr_frame][i][0:3]
            self.spheres[i].local.rotation = quat_swizzle(
                mocap.expmap(glm.vec3(jointpva[self.curr_frame][i][6:9]))
            )

        # update root_group
        root_pos = glm.vec3(self.root[self.curr_frame][3])
        root_rot = quat_swizzle(glm.quat(self.root[self.curr_frame]))
        self.root_group.local.position = root_pos
        self.root_group.local.rotation = root_rot

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

    # unpickle skeleton, xforms, jointpva
    skeleton = mocap.unpickle_obj(outbasepath + "_skeleton.pkl")
    xforms = mocap.unpickle_obj(outbasepath + "_xforms.pkl")
    root = mocap.unpickle_obj(outbasepath + "_root.pkl")
    jointpva = np.load(outbasepath + "_jointpva.npy")

    renderBuddy = RenderBuddy(skeleton, xforms, root, jointpva)
    run()
