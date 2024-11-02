import glm
import pygfx as gfx
from tqdm import trange, tqdm
import sys
from wgpu.gui.auto import WgpuCanvas, run

spheres = []
lines = []

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

playing = True


def animate():
    global frame

    if playing:
        frame = frame + 1
        if frame >= end_frame or frame >= len(g_inputs):
            frame = start_frame

    VEL_SCALE_FACTOR = 0.1
    for j in range(len(g_inputs[frame].j_pos)):
        pos = g_inputs[frame].j_pos[j]
        spheres[j].local.position = pos
        lines[j].local.position = pos
        vel = glm.vec3(g_inputs[frame].j_vel[j])
        vel.z = -vel.z  # AJT HACK, there's some kind of left handed coord thing going on with lines... wtf.
        speed = glm.length(vel)
        if speed > 0.1:
            lines[j].local.rotation = glm.quatLookAt(glm.normalize(vel), glm.vec3(0, 1, 0))
            lines[j].local.scale = speed * VEL_SCALE_FACTOR
        else:
            lines[j].local.rotation = glm.quat()
            lines[j].local.scale = 0.1

    global renderer, canvas, scene, camera, controller

    renderer.render(scene, camera)
    canvas.request_draw()


def on_key_down(event):
    global playing
    if event.key == "Escape":
        renderer.target.close()
    elif event.key == " ":
        playing = not playing


def visualize_input(skeleton, inputs):

    global renderer, canvas, scene, camera, controller

    scene = gfx.Scene()
    scene.add(gfx.AmbientLight(intensity=1))
    scene.add(gfx.DirectionalLight())

    global g_inputs
    g_inputs = inputs

    print("building scene")

    joint_colors = ["#ffffff" for j in range(skeleton.num_joints)]
    joint_colors[skeleton.get_joint_index("Head")] = "#00ff00"
    joint_colors[skeleton.get_joint_index("LeftToeBase")] = "#0000ff"
    joint_colors[skeleton.get_joint_index("RightToeBase")] = "#ff0000"
    joint_colors[skeleton.get_joint_index("LeftForeArm")] = "#ffff00"

    joint_size = [0.5 for j in range(skeleton.num_joints)]
    joint_size[skeleton.get_joint_index("Head")] = 1.5

    # build sphere for every transform
    for j in range(len(inputs[0].j_pos)):
        radius = joint_size[j]
        sphere = gfx.Mesh(
            gfx.sphere_geometry(radius),
            gfx.MeshPhongMaterial(color=joint_colors[j]),
        )
        scene.add(sphere)
        sphere.local.position = inputs[0].j_pos[j]
        spheres.append(sphere)

        line = gfx.Line(
            gfx.Geometry(positions=[[0, 0, 0], [0, 0, -1]]),
            gfx.LineMaterial(thickness=4.0, color="#ff0000"),
        )
        scene.add(line)
        line.local.position = inputs[0].j_pos[j]
        lines.append(line)

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
