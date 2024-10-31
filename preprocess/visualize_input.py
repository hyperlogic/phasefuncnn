import glm
import pygfx as gfx
from tqdm import trange, tqdm


cubes = []
frame = 0
g_inputs = []


def animate():
    global frame
    frame = frame + 1
    if frame >= len(g_inputs):
        frame = 0

    for j in range(len(g_inputs[frame].j_pos)):
        cubes[j].local.position = g_inputs[frame].j_pos[j]



def visualize_input(skeleton, inputs):
    scene = gfx.Scene()
    scene.add(gfx.AmbientLight())
    scene.add(gfx.DirectionalLight())
    camera = gfx.PerspectiveCamera(70, 16 / 9)

    global g_inputs
    g_inputs = inputs

    # add a cube for each j_pos in input
    print("building scene")

    joint_colors = ["#ffffff" for j in range(skeleton.num_joints)]
    joint_colors[skeleton.get_joint_index("Head")] = "#00ff00"
    joint_colors[skeleton.get_joint_index("LeftToeBase")] = "#0000ff"
    joint_colors[skeleton.get_joint_index("RightToeBase")] = "#ff0000"

    # build cube for every transform
    for j in range(len(inputs[0].j_pos)):
        cube = gfx.Mesh(
            gfx.box_geometry(0.5, 0.5, 0.5),
            gfx.MeshPhongMaterial(color=joint_colors[j])
        )
        scene.add(cube)
        cube.local.position = inputs[0].j_pos[j]
        cubes.append(cube)

    AXIS_LEN = 10
    x_axis = gfx.Mesh(gfx.box_geometry(AXIS_LEN, 0.5, 0.5), gfx.MeshPhongMaterial(color="#ff0000"))
    x_axis.local.position = glm.vec3(AXIS_LEN / 2, 0, 0)
    y_axis = gfx.Mesh(gfx.box_geometry(0.5, 10.0, 0.5), gfx.MeshPhongMaterial(color="#00ff00"))
    y_axis.local.position = glm.vec3(0, AXIS_LEN / 2, 0)
    z_axis = gfx.Mesh(gfx.box_geometry(0.5, 0.5, 10.0), gfx.MeshPhongMaterial(color="#0000ff"))
    z_axis.local.position = glm.vec3(0, 0, AXIS_LEN / 2)

    scene.add(x_axis)
    scene.add(y_axis)
    scene.add(z_axis)

    gfx.show(scene, before_render=animate)
