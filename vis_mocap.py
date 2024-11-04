import glm
import mocap
import os
import pickle
import pygfx as gfx
from tqdm import trange, tqdm
import sys
from wgpu.gui.auto import WgpuCanvas, run

OUTPUT_DIR = "output"


# rot from b to a
def quat_dir(a, b):
    dot = glm.dot(b, a)
    cross = glm.cross(b, a)
    return glm.normalize(1 + dot, cross.x, cross.y, cross.z)


class RenderBuddy:
    def __init__(self, skeleton, xforms, root):

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

        # build sphere for every transform
        self.spheres = []
        for j in range(skeleton.num_joints):
            radius = 0.5
            sphere = gfx.Mesh(
                gfx.sphere_geometry(radius), gfx.MeshPhongMaterial(color="#ffffff")
            )
            self.scene.add(sphere)
            sphere.local.position = glm.vec3(xforms[0][j][3])
            self.spheres.append(sphere)

        self.camera = gfx.PerspectiveCamera(70, 4 / 3)
        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.controller = gfx.OrbitController(
            camera=self.camera, register_events=self.renderer
        )

        # draw a disc under the root position.
        self.root_sphere = gfx.Mesh(
            gfx.sphere_geometry(1), gfx.MeshPhongMaterial(color="#aaaaff")
        )
        self.root_sphere.local.position = glm.vec3(self.root[0][3])
        self.root_sphere.local.scale = glm.vec3(2, 0.01, 2)
        self.scene.add(self.root_sphere)

        # draw a line for the root facing dir
        self.root_line = gfx.Line(
            gfx.Geometry(positions=[[0, 0, 0], [-1, 0, 0]]),
            gfx.LineMaterial(thickness=4.0, color="#ff0000"),
        )
        self.root_line.local.position = glm.vec3(self.root[0][3])
        self.root_line.local.scale = 5
        self.root_line.local.rotation = glm.quat(self.root[0])
        self.scene.add(self.root_line)

        self.renderer.add_event_handler(
            lambda event: self.on_key_down(event), "key_down"
        )

        self.canvas.request_draw(lambda: self.animate())

    def animate(self):
        if self.playing:
            self.curr_frame = self.curr_frame + 1
            if self.curr_frame >= self.end_frame or self.curr_frame >= len(self.xforms):
                self.curr_frame = self.start_frame

        for j in range(self.skeleton.num_joints):
            pos = glm.vec3(self.xforms[self.curr_frame][j][3])
            self.spheres[j].local.position = pos

        # update root_sphere
        rpos = glm.vec3(self.root[self.curr_frame][3])
        self.root_sphere.local.position = rpos

        # update root_line
        self.root_line.local.position = rpos
        self.root_line.local.rotation = glm.quat(self.root[self.curr_frame])

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

    # unpickle skeleton, xforms, rpos, rdir
    skeleton = mocap.unpickle_obj(
        os.path.join(OUTPUT_DIR, mocap_basename + "_skeleton.pkl")
    )
    xforms = mocap.unpickle_obj(
        os.path.join(OUTPUT_DIR, mocap_basename + "_xforms.pkl")
    )
    root = mocap.unpickle_obj(os.path.join(OUTPUT_DIR, mocap_basename + "_root.pkl"))

    renderBuddy = RenderBuddy(skeleton, xforms, root)
    run()
