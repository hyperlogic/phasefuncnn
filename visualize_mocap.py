import glm
import mocap
import os
import pickle
import pygfx as gfx
from tqdm import trange, tqdm
import sys
from wgpu.gui.auto import WgpuCanvas, run

OUTPUT_DIR = "output"

start_frame = 0
end_frame = sys.maxsize
curr_frame = start_frame
playing = True
spheres = []


def animate():
    global renderer, canvas, scene, camera, skeleton, xforms, spheres, curr_frame

    if playing:
        curr_frame = curr_frame + 1
        if curr_frame >= end_frame or curr_frame >= len(xforms):
            curr_frame = start_frame

    for j in range(skeleton.num_joints):
        pos = glm.vec3(xforms[curr_frame][j][3])
        spheres[j].local.position = pos

    renderer.render(scene, camera)
    canvas.request_draw()


def on_key_down(event):
    pass


def visualize(skeleton, xforms):
    global renderer, canvas, scene, camera, spheres

    scene = gfx.Scene()
    scene.add(gfx.AmbientLight(intensity=1))
    scene.add(gfx.DirectionalLight())
    scene.add(gfx.helpers.AxesHelper(10.0, 0.5))
    scene.add(gfx.helpers.GridHelper(size=100))

    # build sphere for every transform
    for j in range(skeleton.num_joints):
        radius = 0.5
        sphere = gfx.Mesh(
            gfx.sphere_geometry(radius), gfx.MeshPhongMaterial(color="#ffffff")
        )
        scene.add(sphere)
        sphere.local.position = glm.vec3(xforms[0][j][3])
        spheres.append(sphere)

    camera = gfx.PerspectiveCamera(70, 4 / 3)
    camera.show_object(scene, up=(0, 1, 0), scale=1.4)

    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)
    controller = gfx.OrbitController(camera=camera, register_events=renderer)

    renderer.add_event_handler(on_key_down, "key_down")

    canvas.request_draw(animate)
    run()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]

    global skeleton, xforms

    # unpickle skeleton
    skeleton_filename = os.path.join(OUTPUT_DIR, mocap_basename + "_skeleton.pkl")
    with open(skeleton_filename, "rb") as f:
        skeleton = pickle.load(f)

    # unpickle xforms
    xforms_filename = os.path.join(OUTPUT_DIR, mocap_basename + "_xforms.pkl")
    with open(xforms_filename, "rb") as f:
        xforms = pickle.load(f)

    visualize(skeleton, xforms)
