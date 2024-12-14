import bvh


class Skeleton:
    def __init__(self, bvh):
        self.joint_names = bvh.get_joints_names()
        self.root_name = self.joint_names[0]
        self.joint_index_map = {self.joint_names[i]: i for i in range(len(self.joint_names))}
        self.has_pos_map = {j: "Xposition" in bvh.joint_channels(j) for j in self.joint_names}
        self.has_rot_map = {j: "Xrotation" in bvh.joint_channels(j) for j in self.joint_names}
        self.parent_map = {j: bvh.joint_parent_index(j) for j in self.joint_names}
        self.joint_offset_map = {j: bvh.joint_offset(j) for j in self.joint_names}
        self.num_joints = len(self.joint_names)
        self.build_mirror_map()

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def is_root(self, joint_name):
        return joint_name == self.root_name

    def get_joint_name(self, joint_index):
        return self.joint_names[joint_index]

    def get_joint_index(self, joint_name):
        return self.joint_index_map.get(joint_name, -1)

    def get_parent_index(self, joint_name):
        return self.parent_map[joint_name]

    def has_pos(self, joint_name):
        return self.has_pos_map[joint_name]

    def has_rot(self, joint_name):
        return self.has_rot_map[joint_name]

    def get_joint_offset(self, joint_name):
        return self.joint_offset_map[joint_name]

    def build_mirror_map(self):
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
