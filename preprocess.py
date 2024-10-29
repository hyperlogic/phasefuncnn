import bvhio

# Reads the file into a transform tree structure and converts all data to build proper local and world spaces.
# This structure allows for extensive read of properties and spaces and does also support modifications of the animation.
root = bvhio.readAsHierarchy('../PFNN/data/animations/LocomotionFlat01_000.bvh')
root.printTree()
