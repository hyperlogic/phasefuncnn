import bvh


class Skeleton:
    joint_names: list[str]
    root_name: str
    joint_index_map: dict[str, int]
    has_pos_map: dict[str, bool]
    has_rot_map: dict[str, bool]
    parent_map: dict[str, int]
    child_map: dict[str, list[int]]
    joint_offset_map: dict[str, list[float]]
    num_joints: int

    def __init__(self, bvh: bvh.Bvh):
        self.joint_names = bvh.get_joints_names()
        self.root_name = self.joint_names[0]
        self.joint_index_map = {self.joint_names[i]: i for i in range(len(self.joint_names))}
        self.has_pos_map = {j: "Xposition" in bvh.joint_channels(j) for j in self.joint_names}
        self.has_rot_map = {j: "Xrotation" in bvh.joint_channels(j) for j in self.joint_names}
        self.parent_map = {j: bvh.joint_parent_index(j) for j in self.joint_names}
        self.joint_offset_map = {j: list(bvh.joint_offset(j)) for j in self.joint_names}
        self.num_joints = len(self.joint_names)
        self._build_mirror_map()
        self._build_child_map()

    def is_root(self, joint_name: str) -> bool:
        return joint_name == self.root_name

    def get_joint_name(self, joint_index: int) -> str:
        return self.joint_names[joint_index]

    def get_joint_index(self, joint_name: str) -> int:
        return self.joint_index_map.get(joint_name, -1)

    def get_parent_index(self, joint_name: str) -> int:
        return self.parent_map[joint_name]

    def get_children_indices(self, joint_name: str) -> list[int]:
        return self.child_map[joint_name]

    def has_pos(self, joint_name: str) -> bool:
        return self.has_pos_map[joint_name]

    def has_rot(self, joint_name: str) -> bool:
        return self.has_rot_map[joint_name]

    def get_joint_offset(self, joint_name: str) -> list[float]:
        return self.joint_offset_map[joint_name]

    def _build_mirror_map(self):
        self.mirror_map = []
        center_joints = ["Hips", "LowerBack", "Spine", "Spine1", "Neck", "Head"]
        for i in range(self.num_joints):
            name = self.joint_names[i]
            mirror_name = ""
            mirror_index = -1
            if name in center_joints:
                pass
            elif name.startswith("Left"):
                mirror_name = name.replace("Left", "Right", 1)
                mirror_index = self.get_joint_index(mirror_name)
            elif name.startswith("Right"):
                mirror_name = name.replace("Right", "Left", 1)
                mirror_index = self.get_joint_index(mirror_name)
            elif name.startswith("L"):
                mirror_name = name.replace("L", "R", 1)
                mirror_index = self.get_joint_index(mirror_name)
            elif name.startswith("R"):
                mirror_name = name.replace("R", "L", 1)
                mirror_index = self.get_joint_index(mirror_name)

            if mirror_index >= 0:
                self.mirror_map.append(mirror_index)
            else:
                self.mirror_map.append(i)

    def _build_child_map(self):
        self.child_map = {name: [] for name in self.joint_names}
        for child_name in self.joint_names:
            child_index = self.get_joint_index(child_name)
            parent_index = self.get_parent_index(child_name)
            if parent_index >= 0:
                parent_name = self.get_joint_name(parent_index)
                self.child_map[parent_name].append(child_index)

