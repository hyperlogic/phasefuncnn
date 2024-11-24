#
#
#

import os
import time
from pathlib import Path

import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run

model_dir = Path(os.getcwd()) / "data"
gltf_path = model_dir / "xbot-walk.glb"

canvas = WgpuCanvas(size=(640, 480), max_fps=-1, title="Skinnedmesh", vsync=False)

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
camera.local.position = (0, 100, 200)
camera.look_at((0, 100, 0))
scene = gfx.Scene()

scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

gltf = gfx.load_gltf(gltf_path, quiet=True)

# gfx.print_tree(gltf.scene) # Uncomment to see the tree structure

model_obj = gltf.scene.children[0]
model_obj.local.scale = (1, 1, 1)

current_clip = 0
anim = gltf.animations[0]

skeleton_helper = gfx.SkeletonHelper(model_obj)
scene.add(skeleton_helper)
scene.add(model_obj)

gfx.OrbitController(camera, register_events=renderer)

global_time = 0.0
last_time = time.perf_counter()
global_frame = 0
num_frames = len(anim['tracks'][0]['times'])

stats = gfx.Stats(viewport=renderer)

def copy_anim_to_skeleton(frame, anim, skeleton):
    # TODO: do linear interpolation of keyframes
    for track in anim['tracks']:
        property = track['property']
        times = track['times']
        target = track['target']
        values = track['values']
        if frame < len(values):
            if property == 'rotation':
                target.local.rotation = values[frame]
            elif property == 'translation':
                target.local.position = values[frame]
            elif property == 'scale':
                target.local.scale = values[frame]
        #else:
            #print(f"bad frame {frame} for track {track['name']}, len = {len(values)}, times = {times}")

#s = gfx.Skeleton([], [])
#s.bones[0].local.position = xx

def animate():
    global global_time, last_time, anim, global_frame, num_frames
    now = time.perf_counter()
    dt = now - last_time
    last_time = now
    global_time += dt
    if global_time > anim["duration"]:
        global_time = 0

    global_frame += 1
    if global_frame > num_frames:
        global_frame = 0

    copy_anim_to_skeleton(global_frame, anim, model_obj.children[0].skeleton)

    model_obj.children[0].skeleton.update()
    skeleton_helper.update()

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
