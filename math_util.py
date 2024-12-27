import math
import numpy as np
import pickle

DEADSPOT_THRESH = 0.15


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / np.linalg.norm(v)
    else:
        return np.array([1, 0, 0])


def deadspot(val: float) -> float:
    if np.abs(val) > DEADSPOT_THRESH:
        return val
    else:
        return 0


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
    return mat


def build_mat_rotx(mat, alpha):
    cosa, sina = math.cos(alpha), math.sin(alpha)
    mat[0] = [1, 0, 0, 0]
    mat[1] = [0, cosa, -sina, 0]
    mat[2] = [0, sina, cosa, 0]
    mat[3] = [0, 0, 0, 1]
    return mat


def build_mat_roty(mat, beta):
    cosb, sinb = math.cos(beta), math.sin(beta)
    mat[0] = [cosb, 0, sinb, 0]
    mat[1] = [0, 1, 0, 0]
    mat[2] = [-sinb, 0, cosb, 0]
    mat[3] = [0, 0, 0, 1]
    return mat


def build_mat_rotz(mat, gamma):
    cosg, sing = math.cos(gamma), math.sin(gamma)
    mat[0] = [cosg, -sing, 0, 0]
    mat[1] = [sing, cosg, 0, 0]
    mat[2] = [0, 0, 1, 0]
    mat[3] = [0, 0, 0, 1]
    return mat


def build_mat_from_quat(mat, quat):
    x, y, z, w = quat

    # Compute the matrix elements
    mat[0] = [
        1 - 2 * y**2 - 2 * z**2,
        2 * x * y - 2 * z * w,
        2 * x * z + 2 * y * w,
        0,
    ]
    mat[1] = [
        2 * x * y + 2 * z * w,
        1 - 2 * x**2 - 2 * z**2,
        2 * y * z - 2 * x * w,
        0,
    ]
    mat[2] = [
        2 * x * z - 2 * y * w,
        2 * y * z + 2 * x * w,
        1 - 2 * x**2 - 2 * y**2,
        0,
    ]
    mat[3] = [0, 0, 0, 1]
    return mat


def quat_from_mat(mat):
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


def quat_from_vectors(from_vec, to_vec):
    # Normalize the input vectors
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)

    # Compute the cross product and the angle between the vectors
    cross_prod = np.cross(from_vec, to_vec)
    dot_prod = np.dot(from_vec, to_vec)

    # Calculate the scalar part of the quaternion
    w = np.sqrt(np.linalg.norm(from_vec) ** 2 * np.linalg.norm(to_vec) ** 2) + dot_prod

    # Handle cases where vectors are parallel or anti-parallel
    if np.isclose(w, 0.0):  # Vectors are opposite
        # Rotate 180 degrees around an arbitrary axis perpendicular to `from_vec`
        # We can use the x-axis [1,0,0] if from_vec isn't aligned with it,
        # otherwise, we use the y-axis [0,1,0].
        if np.abs(from_vec[0]) < 1.0:
            axis = np.array([1, 0, 0])
        else:
            axis = np.array([0, 1, 0])
        cross_prod = np.cross(from_vec, axis)
        w = 0.0

    quat = np.array([cross_prod[0], cross_prod[1], cross_prod[2], w])

    # Normalize the quaternion
    return quat / np.linalg.norm(quat)


def quat_mirror(quat):
    return np.array([quat[0], -quat[1], -quat[2], quat[3]])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    quaternion multiplication
    """
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([x, y, z, w])


def quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vq = np.array([vector[0], vector[1], vector[2], 0])
    return quat_mul(quat_mul(q, vq), quat_conj(q))[0:3]


def quat_from_angle_axis(theta: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    if np.abs(theta) < 1e-6:
        return np.array([0, 0, 0, 1])
    half_theta = theta * 0.5
    img = math.sin(half_theta) * axis
    return np.array([img[0], img[1], img[2], math.cos(half_theta)])


def mat_mirror(m):
    mm = np.eye(4)
    build_mat_from_quat(mm, quat_mirror(quat_from_mat(m)))
    mm[0:3, 3] = m[0:3, 3]
    mm[0, 3] = -m[0, 3]
    return mm


def orthogonalize_camera_mat(z: np.ndarray, y: np.ndarray, pos: np.ndarray) -> np.ndarray:
    # make sure that camera_mat will be orthogonal, and aligned with world up (y).
    camera_mat = np.eye(4)
    if np.dot(z, y) < 0.999:  # if w are aren't looking stright up.
        xx = normalize(np.linalg.cross(y, z))
        yy = normalize(np.linalg.cross(z, xx))
        camera_mat[:3, 0] = xx
        camera_mat[:3, 1] = yy
        camera_mat[:3, 2] = z
        camera_mat[:3, 3] = pos
    else:
        camera_mat[:3, 3] = pos
    return camera_mat


def build_look_at_mat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = -normalize(target - eye)
    camera_mat = orthogonalize_camera_mat(z, up, eye)
    return camera_mat
