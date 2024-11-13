import glm
import math
import numpy as np
import pickle


def pickle_obj(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def expmap(v):
    """convert a rotation expressed in r^3 to a unit quaternion"""
    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return np.array([0, 0, 0, 1])
    half_theta = theta * 0.5
    img = (math.sin(half_theta) / theta) * v
    return np.array([img[0], img[1], img[2], math.cos(half_theta)])


def logmap(q):
    """convert a quaternion into a rotation expressed in r^3"""

    quat = q / np.linalg.norm(q)
    angle = 2 * math.acos(quat[3])

    # Avoid division by zero when angle is very small
    img_len = np.linalg.norm(quat[0:3])
    if img_len < 1e-6:
        return np.array([0, 0, 0])
    axis = quat[0:3] / img_len
    return angle * axis


# alpha - rotation about x axis
# beta - rotaiton about y axis
# gamma - rotaiton about z axis
# mat = rotz @ roty @ rotx
def build_mat_from_euler(mat, alpha, beta, gamma):
    cosa, sina = math.cos(alpha), math.sin(alpha)
    cosb, sinb = math.cos(beta), math.sin(beta)
    cosg, sing = math.cos(gamma), math.sin(gamma)

    mat[0] = [
        cosb * cosg,
        cosg * sina * sinb - cosa * sing,
        cosa * cosg * sinb + sina * sing,
        0,
    ]
    mat[1] = [
        cosb * sing,
        cosa * cosg + sina * sinb * sing,
        -cosg * sina + cosa * sinb * sing,
        0,
    ]
    mat[2] = [-sinb, cosb * sina, cosa * cosb, 0]
    mat[3] = [0, 0, 0, 1]


def build_mat_rotx(mat, alpha):
    cosa, sina = math.cos(alpha), math.sin(alpha)
    mat[0] = [1, 0, 0, 0]
    mat[1] = [0, cosa, -sina, 0]
    mat[2] = [0, sina, cosa, 0]
    mat[3] = [0, 0, 0, 1]


def build_mat_roty(mat, beta):
    cosb, sinb = math.cos(beta), math.sin(beta)
    mat[0] = [cosb, 0, sinb, 0]
    mat[1] = [0, 1, 0, 0]
    mat[2] = [-sinb, 0, cosb, 0]
    mat[3] = [0, 0, 0, 1]


def build_mat_rotz(mat, gamma):
    cosg, sing = math.cos(gamma), math.sin(gamma)
    mat[0] = [cosg, -sing, 0, 0]
    mat[1] = [sing, cosg, 0, 0]
    mat[2] = [0, 0, 1, 0]
    mat[3] = [0, 0, 0, 1]


def build_mat_from_quat(quat):
    w, x, y, z = quat

    # Compute the matrix elements
    matrix = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w, 0],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2, 0],
            [0, 0, 0, 1],
        ]
    )

    return matrix


def build_quat_from_mat(mat):
    # Extract the rotation part of the matrix (top-left 3x3)
    m = mat[:3, :3]

    # Calculate the trace of the matrix
    trace = np.trace(m)

    if trace > 0:
        # For a positive trace
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        # For a non-positive trace, find the largest diagonal element
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    return np.array([x, y, z, w])
