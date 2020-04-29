"""Module containing robotics related functions and classes.

Author: Sivakumar Balasubramanian
Date: 24 May 2018
"""

import numpy as np


def rotx(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, +ct, -st],
                     [0.0, +st, +ct]])


def roty(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[+ct, 0.0, +st],
                     [0.0, 1.0, 0.0],
                     [-st, 0.0, +ct]])


def rotz(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[+ct, -st, 0.0],
                     [+st, +ct, 0.0],
                     [0.0, 0.0, 1.0]])


def HTMat(R, d):
    _R = np.hstack((R, d))
    return np.vstack((_R, np.array([[0, 0, 0, 1]])))


def HTMat4DH(t, d, a, al):
    _Hx = HTMat(rotz(t), np.zeros((3, 1)))
    _Hd = HTMat(np.eye(3), np.array([[0, 0, d]]).T)
    _Ha = HTMat(np.eye(3), np.array([[a, 0, 0]]).T)
    _Hal = HTMat(rotx(al), np.zeros((3, 1)))
    return reduce(np.dot, [_Hx, _Hd, _Ha, _Hal])


def forward_kinematics(dhparam, t):
    """Returns the location and orientation of the different frames with 
    respect to the base frame the given configuation.
    """
    if len(t) != len(dhparam):
        _str = " ".join(("Error! No. angles must equal", 
                         "the no. of DOFs of the robot."))
        raise ValueError(_str)
        return None

    _H = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
          for _t, _dh in zip(t, dhparam)]
    H = [_H[0]]
    for i in xrange(1, len(_H)):
        H.append(np.matmul(H[i-1], _H[i]))
    return H, _H


def vec_to_skewsymmat(v):
    """Returns a skew symmetric matrix for the given 
    vector.
    """
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],    0]])


def rotmat_to_eulerparam(rotm):
    """Converts the given rotation matrix into Euler parameters.
    """
    # Angle of rotation
    phi = np.arccos(0.5 * (np.trace(rotm) - 1))

    # Axis of rotation
    if np.isclose(np.sin(phi), 0.0):
        ep = np.array([1, 0, 0, 0])
        return ep
    else:
        a = np.array([rotm[2, 1] - rotm[1, 2],
                      rotm[0, 2] - rotm[2, 0],
                      rotm[1, 0] - rotm[0, 1]]) * (0.5 / np.sin(phi))
        ep = np.hstack((np.cos(0.5 * phi), np.sin(0.5 * phi) * a))    
        return ep


def eulerparam_to_rotmat(ep):
    """Converts the given Euler parameters into a
    rotation matrix.
    """
    eta = ep[0]
    eps = np.array([ep[1:4]]).T
    _r1 = (eta ** 2 - np.matmul(eps.T, eps)[0, 0]) * np.eye(3)
    _r2 = 2 * np.dot(eps, eps.T)
    _r3 = 2 * eta * vec_to_skewsymmat(eps.T[0])
    return _r1 + _r2 + _r3


def euleraram_to_angaxis(ep):
    """Converts the given Euler parameters into angle-axis 
    parameters.
    """
    # Angle of rotation.
    phi = 2 * np.arccos(ep[0])
    
    # Check if the norm of epsilon is zero.
    if np.linalg.norm(ep[1:4]) == 0:
        a = np.array([1, 0, 0])
    elif np.sin(0.5 * phi) == 0:
        a = np.array([1, 0, 0])
    else:
        a = ep[1:4] / np.sin(0.5 * phi)
    
    return np.hstack(([phi], a))
