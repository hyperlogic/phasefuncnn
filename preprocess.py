import bvhio

# Reads the file into a transform tree structure and converts all data to build proper local and world spaces.
# This structure allows for extensive read of properties and spaces and does also support modifications of the animation.
# root = bvhio.readAsHierarchy('../PFNN/data/animations/LocomotionFlat09_000.bvh')
# root.printTree()

bvh = bvhio.readAsBvh("../PFNN/data/animations/LocomotionFlat09_000.bvh")

print(f"Root: {bvh.Root}")
print(f"Frames: {bvh.FrameCount}")
print(f"Frame time: {bvh.FrameTime}")

hierarchy = bvhio.readAsHierarchy("../PFNN/data/animations/LocomotionFlat09_000.bvh")

# joints from the hierarchy can be selected by their name
joint = hierarchy.filter("Hips")[0]

print(f"hips num keyframes = {len(joint.Keyframes)}")
