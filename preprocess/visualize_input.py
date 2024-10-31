import glm
import pygfx as gfx
from tqdm import trange, tqdm


def animate():
    pass


def visualize_input(input):
    scene = gfx.Scene()
    scene.add(gfx.AmbientLight())
    scene.add(gfx.DirectionalLight())
    camera = gfx.PerspectiveCamera(70, 16 / 9)

    # add a cube for each j_pos in input
    cubes = []
    print("building scene")
    for i in trange(input.j_pos.shape[0]):
        p = input.j_pos[i]
        cube = gfx.Mesh(
            gfx.box_geometry(0.1, 0.1, 0.1), gfx.MeshPhongMaterial(color="#ff6699")
        )
        scene.add(cube)
        cube.local.position = p
        cubes.append(cube)

    gfx.show(scene, before_render=animate)
