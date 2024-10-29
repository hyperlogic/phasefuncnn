from bvh import Bvh
import glm
import pygfx as gfx

# Reads the file into a transform tree structure and converts all data to build proper local and world spaces.
# This structure allows for extensive read of properties and spaces and does also support modifications of the animation.
# root = bvhio.readAsHierarchy('../PFNN/data/animations/LocomotionFlat09_000.bvh')
# root.printTree()

with open("../PFNN/data/animations/LocomotionFlat09_000.bvh") as f:
    bvh = Bvh(f.read())

print(f"frames = {bvh.nframes}")
print(f"frame_time = {bvh.frame_time}")

# get root
root = next(bvh.root.filter("ROOT"))
root_name = root.name
root_offset = bvh.joint_offset(root_name)
root_channels = bvh.joint_channels(root_name)

# extract Hips positions
root_positions = []
for i in range(bvh.nframes):
    x_pos = bvh.frame_joint_channel(i, root_name, 'Xposition')
    y_pos = bvh.frame_joint_channel(i, root_name, 'Yposition')
    z_pos = bvh.frame_joint_channel(i, root_name, 'Zposition')
    root_positions.append([x_pos, y_pos, z_pos])

# print offsets
print(f"root_offset = {root_offset}")
print(f"root_channels = {root_channels}")

scene = gfx.Scene()
scene.add(gfx.AmbientLight())
scene.add(gfx.DirectionalLight())
camera = gfx.PerspectiveCamera(70, 16 / 9)

# add a cube for each frame of Hips data
cubes = []
for p in root_positions:
    cube = gfx.Mesh(gfx.box_geometry(0.1, 0.1, 0.1),
                    gfx.MeshPhongMaterial(color="#ff6699"))
    scene.add(cube)
    cube.local.position = p
    cubes.append(cube)

def animate():
    pass

if __name__ == "__main__":
    gfx.show(scene, before_render=animate)
