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



def visualize_input(inputs):
    scene = gfx.Scene()
    scene.add(gfx.AmbientLight())
    scene.add(gfx.DirectionalLight())
    camera = gfx.PerspectiveCamera(70, 16 / 9)

    global g_inputs
    g_inputs = inputs

    # add a cube for each j_pos in input
    print("building scene")

    # build cube for every transform
    for j in range(len(inputs[0].j_pos)):
        cube = gfx.Mesh(
            gfx.box_geometry(0.5, 0.5, 0.5), gfx.MeshPhongMaterial(color="#ff6699")
        )
        scene.add(cube)
        cube.local.position = inputs[0].j_pos[j]
        cubes.append(cube)

    gfx.show(scene, before_render=animate)
