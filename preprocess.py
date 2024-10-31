from bvh import Bvh
import glm
import preprocess

with open("../PFNN/data/animations/LocomotionFlat09_000.bvh") as f:
    bvh = Bvh(f.read())

print(f"frames = {bvh.nframes}")
print(f"frame_time = {bvh.frame_time}")

skeleton, input = preprocess.build_input(bvh)

if __name__ == "__main__":
    preprocess.visualize_input(skeleton, input)
