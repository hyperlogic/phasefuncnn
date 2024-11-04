import glm
import math
import pickle


def pickle_obj(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def expmap(v):
    """convert a rotation expressed in r^3 to a unit quaternion"""
    theta = glm.len(v)
    if theta < 1e-6:
        return glm.quat()
    half_theta = theta * 0.5
    img = (math.sin(half_theta) / theta) * v
    return glm.quat(math.cos(half_theta), img)


def logmap(q):
    """convert a quaternion into a rotation expressed in r^3"""
    quat = glm.normalize(q)
    angle = 2 * math.acos(quat.w)
    # Avoid division by zero when angle is very small
    sin_half_angle = math.sqrt(1 - (quat.w * quat.w))
    if sin_half_angle < 1e-6:
        return glm.vec3(0, 0, 0)
    axis = glm.vec3(quat.x, quat.y, quat.z) / sin_half_angle
    return (angle / 2) * axis
