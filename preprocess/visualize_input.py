import glm
import pygfx as gfx
from tqdm import trange, tqdm
import sys
from wgpu.gui.auto import WgpuCanvas, run

cubes = []

# start_frame = 156
# end_frame = 158
start_frame = 0
end_frame = sys.maxsize

frame = start_frame

g_inputs = []

scene = None
renderer = None
camera = None
canvas = None
controller = None


def animate():
    global frame
    frame = frame + 1
    if frame >= end_frame or frame >= len(g_inputs):
        frame = start_frame

    for j in range(len(g_inputs[frame].j_pos)):
        cubes[j].local.position = g_inputs[frame].j_pos[j]

    global renderer, canvas, scene, camera, controller

    renderer.render(scene, camera)
    canvas.request_draw()


def on_key_down(event):
    if event.key == "Escape":
        renderer.target.close()


def visualize_input(skeleton, inputs):

    global renderer, canvas, scene, camera, controller

    scene = gfx.Scene()
    scene.add(gfx.AmbientLight(intensity=1))
    scene.add(gfx.DirectionalLight())

    global g_inputs
    g_inputs = inputs

    # add a cube for each j_pos in input
    print("building scene")

    joint_colors = ["#ffffff" for j in range(skeleton.num_joints)]
    joint_colors[skeleton.get_joint_index("Head")] = "#00ff00"
    joint_colors[skeleton.get_joint_index("LeftToeBase")] = "#0000ff"
    joint_colors[skeleton.get_joint_index("RightToeBase")] = "#ff0000"
    joint_colors[skeleton.get_joint_index("LeftForeArm")] = "#ffff00"

    joint_size = [0.5 for j in range(skeleton.num_joints)]
    joint_size[skeleton.get_joint_index("Head")] = 3

    # build cube for every transform
    for j in range(len(inputs[0].j_pos)):
        box_size = joint_size[j]
        cube = gfx.Mesh(
            gfx.box_geometry(box_size, box_size, box_size),
            gfx.MeshPhongMaterial(color=joint_colors[j]),
        )
        scene.add(cube)
        cube.local.position = inputs[0].j_pos[j]
        cubes.append(cube)

    scene.add(gfx.helpers.AxesHelper(10.0, 0.5))
    scene.add(gfx.helpers.GridHelper(size=100))

    camera = gfx.PerspectiveCamera(70, 4 / 3)
    camera.show_object(scene, up=(0, 1, 0), scale=1.4)

    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)
    controller = gfx.OrbitController(camera=camera, register_events=renderer)

    renderer.add_event_handler(on_key_down, "key_down")

    canvas.request_draw(animate)
    run()
