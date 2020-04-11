import numpy as np

"""Rotation representation
    'angle' x-y-z order Euler angle
            [alpha, beta, gamma]
    'Quaternion' Hamilton‘s quaternion complex w+xi+yj+zk
            [w, x, y ,z]
    'vector' rotation vector
            [rx, ry, rz]
    'matrix' rotation matrix
            [[1, 0, 0, 0],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1]],
            [[cos(beta), 0, sin(beta), 0],
            [0, 1, 0, 0],
            [-sin(beta), 0, cos(beta), 0],
            [0, 0, 0, 1]],
            [[cos(gamma), -sin(gamma), 0, 0],
            [sin(gamma), cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]       
"""


def quaternionMul(quaternion1, quaternion2):
    """Quaternion multiplication
        'quaternion1' [w1, x1, y1 ,z1]
        'quaternion2' [w2, x2, y2 ,z2]
    """
    # quaternion
    w1 = quaternion1[0]
    x1 = quaternion1[1]
    y1 = quaternion1[2]
    z1 = quaternion1[3]
    w2 = quaternion2[0]
    x2 = quaternion2[1]
    y2 = quaternion2[2]
    z2 = quaternion2[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=float)


def getmax(x, y):
    x = float(x)
    y = float(y)
    if x < y:
        x = y
    return x

def getmin(x, y):
    x = float(x)
    y = float(y)
    if x > y:
        x = y
    return x


def sign(x):
    x = float(x)
    s = 0
    if x > 4e-6:
        s = 1
    elif x < -4e-6:
        s = -1
    return s


def signDictionary():
    """xy yz xz sign case looking dictionary"""
    dict = {}
    xs = [-1, 0, 1]
    ys = [-1, 0, 1]
    zs = [-1, 0, 1]
    for x in xs:
        for y in ys:
            for z in zs:
                xy = x * y
                yz = y * z
                xz = x * z
                xx = x * x
                yy = y * y
                zz = z * z
                dict[str(xy + 1) + str(yz + 1) + str(xz + 1) +
                     str(xx + 1) + str(yy + 1) + str(zz + 1)] = [x, y, z]
    return dict


signDict = signDictionary()


def angleToMatrix(angle):
    """Euler angle(x-y-z order) to rotation matrix transform"""
    # angle
    angle = np.array(angle[0:3], dtype=float)
    # matrix
    rotx = np.array([[1, 0, 0, 0],
                     [0, np.cos(angle[0]), -np.sin(angle[0]), 0],
                     [0, np.sin(angle[0]), np.cos(angle[0]), 0],
                     [0, 0, 0, 1]], dtype=float)
    roty = np.array([[np.cos(angle[1]), 0, np.sin(angle[1]), 0],
                     [0, 1, 0, 0],
                     [-np.sin(angle[1]), 0, np.cos(angle[1]), 0],
                     [0, 0, 0, 1]], dtype=float)
    rotz = np.array([[np.cos(angle[2]), -np.sin(angle[2]), 0, 0],
                     [np.sin(angle[2]), np.cos(angle[2]), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)
    matrix = np.dot(np.dot(rotz, roty), rotx)
    # ca = np.cos(angle[0])
    # sa = np.sin(angle[0])
    # cb = np.cos(angle[1])
    # sb = np.sin(angle[1])
    # cc = np.cos(angle[2])
    # sc = np.sin(angle[2])
    # matrix_abc = np.array([[cc*cb, cc*sb*sa-sc*ca, cc*sb*ca+sc*sa, 0],
    #                        [sc*cb, sc*sb*sa+cc*ca, sc*sb*ca-cc*sa, 0],
    #                        [-sb, cb*sa, cb*ca, 0],
    #                        [0, 0, 0, 1]], dtype=float)
    return matrix


def matrixToAngle(matrix):
    """Rotation matrix to Euler angle(x-y-z order) transform"""
    # matrix
    matrix = np.array(matrix[0:4, 0:4], dtype=float)
    # angle
    sb = -matrix[2, 0]
    cb = np.sqrt(matrix[2, 1] * matrix[2, 1] + matrix[2, 2] * matrix[2, 2])
    if cb > 1e-6:
        alpha = np.arctan2(matrix[2, 1], matrix[2, 2])
        beta = np.arctan2(sb, cb)
        gamma = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        alpha = 0.0
        beta = np.arctan2(sb, cb)
        gamma = -np.arctan2(matrix[0, 1], matrix[0, 2])
    angle = np.array([alpha, beta, gamma], dtype=float)
    return angle


def quaternionToMatrix(quaternion):
    """Quaternion to rotation matrix transform"""
    # quaternion
    quaternion = np.array(quaternion[0:4], dtype=float)
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    # matrix
    matrix = np.array([[w * w + x * x - y * y - z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
                       [2 * x * y + 2 * z * w, w * w - x * x + y * y - z * z, 2 * y * z - 2 * x * w, 0],
                       [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, w * w - x * x - y * y + z * z, 0],
                       [0, 0, 0, 1]], dtype=float)
    return matrix


def matrixToQuaternion(matrix):
    """Rotation matrix to quaternion transform"""
    # matrix
    matrix = np.array(matrix[0:4, 0:4], dtype=float)
    # quaternion
    ww = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2] + 1) / 4
    w = np.sqrt(ww)
    if ww > 1e-6:
        x = (matrix[2, 1] - matrix[1, 2]) / (4 * w)
        y = (matrix[0, 2] - matrix[2, 0]) / (4 * w)
        z = (matrix[1, 0] - matrix[0, 1]) / (4 * w)
    else:
        xy = int(sign((matrix[0, 1] + matrix[1, 0])))
        yz = int(sign((matrix[1, 2] + matrix[2, 1])))
        xz = int(sign((matrix[0, 2] + matrix[2, 0])))
        xx = int(sign((matrix[0, 0] + 1) - 2 * ww))
        yy = int(sign((matrix[1, 1] + 1) - 2 * ww))
        zz = int(sign((matrix[2, 2] + 1) - 2 * ww))
        signs = signDict[str(xy + 1) + str(yz + 1) + str(xz + 1) +
                         str(xx + 1) + str(yy + 1) + str(zz + 1)]
        xx = (matrix[0, 0] + 1) / 2 - w * w
        yy = (matrix[1, 1] + 1) / 2 - w * w
        zz = (matrix[2, 2] + 1) / 2 - w * w
        x = signs[0] * np.sqrt(xx)
        y = signs[1] * np.sqrt(yy)
        z = signs[2] * np.sqrt(zz)
    quaternion = np.array([w, x, y, z], dtype=float)
    return quaternion


def vectorToMatrix(vector):
    """Rotation vector to rotation matrix (Rodrigues) transform"""
    # vector
    r = np.array(vector[0:3], dtype=float)
    rlen = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    rn = r / rlen
    # matrix ---- Rodrigues transform
    K = np.array([[0, -rn[2], rn[1], 0],
                  [rn[2], 0, -rn[0], 0],
                  [-rn[1], rn[0], 0, 0],
                  [0, 0, 0, 1]], dtype=float)
    matrix = np.eye(4, dtype=float) + (1 - np.cos(rlen)) * np.dot(K, K) + \
             np.sin(rlen) * np.array([[0, -rn[2], rn[1], 0],
                                      [rn[2], 0, -rn[0], 0],
                                      [-rn[1], rn[0], 0, 0],
                                      [0, 0, 0, 1]], dtype=float)
    return matrix


def matrixToVector(matrix):
    """Rotation matrix to rotation vector (inv Rodrigues) transform"""
    # matrix
    matrix = np.array(matrix[0:4, 0:4], dtype=float)
    # vector
    vector = np.array([0, 0, 0], dtype=float)
    K_ = (matrix - np.transpose(matrix)) / 2
    print(K_)
    srn = np.array([K_[2, 1], K_[0, 2], K_[1, 0]], dtype=float)
    sth_sth = np.sum(np.multiply(srn, srn))
    if sth_sth > 1e-6:
        sth = np.sqrt(sth_sth)
        theta = np.arcsin(sth)
        rn = srn / sth
        vector = theta * rn
    return vector


def angleToQuaternion(angle):
    """Euler angle(x-y-z order) to quaternion transform"""
    # angle
    angle = np.array(angle[0:3], dtype=float)
    # quaternion
    cx = np.cos(angle[0] / 2)
    sx = np.sin(angle[0] / 2)
    cy = np.cos(angle[1] / 2)
    sy = np.sin(angle[1] / 2)
    cz = np.cos(angle[2] / 2)
    sz = np.sin(angle[2] / 2)
    qx = np.array([cx, sx, 0, 0], dtype=float)
    qy = np.array([cy, 0, sy, 0], dtype=float)
    qz = np.array([cz, 0, 0, sz], dtype=float)
    quaternion = quaternionMul(quaternionMul(qx, qy), qz)
    quaternion_xyz = np.array([cx * cy * cz - sx * sy * sz,
                               sx * cy * cz + cx * sy * sz,
                               cx * sy * cz - sx * cy * sz,
                               sx * sy * cz + cx * cy * sz], dtype=float)
    return quaternion


def quaternionToAngle(quaternion):
    """Quaternion to Euler angle(x-y-z order) transform"""
    # quaternion
    quaternion = np.array(quaternion[0:4], dtype=float)
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    # angle
    sbeta = 2 * w * y + 2 * x * z
    if sbeta-1 >= -1e-6:
        beta = np.pi/2
    elif sbeta+1 <= 1e-6:
        beta = -np.pi/2
    else:
        beta = np.arcsin(sbeta)
    alpha = np.arctan2(2 * w * x - 2 * y * z, 1 - 2 * x * x - 2 * y * y)
    gamma = np.arctan2(2 * w * z - 2 * x * y, 1 - 2 * y * y - 2 * z * z)
    angle = np.array([alpha, beta, gamma], dtype=float)
    return angle



def quaternionToVector(quaternion):
    """Quaternion to rotation vector transform"""
    # quaternion
    quaternion = np.array(quaternion[0:4], dtype=float)
    w = quaternion[0]
    # vector
    theta = 2*np.arccos(w)
    stheta = np.sqrt(1-w*w)
    r = quaternion[1:3]
    r = r*theta/stheta
    return r


def vectorToQuaternion(vector):
    """Rotation vector to quaternion transform"""
    # vector
    r = np.array(vector[0:3], dtype=float)
    # quaternion
    theta = np.sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
    w = np.cos(theta/2)
    x = np.sin(theta/2)*r[0]
    y = np.sin(theta/2)*r[1]
    z = np.sin(theta/2)*r[2]
    quaternion = np.array([w, x, y, z], dtype=float)
    return quaternion