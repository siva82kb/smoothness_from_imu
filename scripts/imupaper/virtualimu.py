"""Module containing a list of funtions to simulate a 3DOF arm, with a
virutal IMU at its endpoint.

Author: Sivakumar Balasubramanian
Date: 20 Nov 2018
"""

import numpy as np
import json
import sys

sys.path.append("../scripts/")
import myrobotics as myrob


def harm_inv_kinematics(Xh, L1, L2):
    l_max = np.max([L1, L2])
    l_min = np.max([L1, L2])
    r = np.linalg.norm(Xh)

    # Make sure the given point is within the robot's workspace.
    if r > (l_max + l_min) or r < (l_max - l_min):
        return np.nan, np.nan, np.nan

    # Inverse Kinematics
    # Step 1. Find the t1 using x and y coordinates.
    ta1 = np.arctan2(Xh[1], Xh[0])

    # Step 2. Solve for standard 2DOF arm for t2 and t3.
    _xd = np.sqrt(np.power(Xh[1], 2) + np.power(Xh[0], 2))
    _yd = Xh[2]
    _n = _xd**2 + _yd**2 - L1**2 - L2**2
    _d = 2 * L1 * L2
    ta3 = np.arccos(np.min([1.0, _n / _d]))

    alpha = np.arctan2(_yd, _xd)

    _n = _xd**2 + _yd**2 + L1**2 - L2**2
    _d = 2 * L1 * np.sqrt(_xd**2 + _yd**2)
    beta = np.arccos(np.min([1.0, _n / _d]))

    ta2 = alpha - beta

    return ta1, ta2, ta3


def mjt(t):
    """Return MJT position trajectory.
    """
    np.clip(t, 0, 1.0, out=t)
    return 10 * np.power(t, 3) - 15 * np.power(t, 4) + 6 * np.power(t, 5)


def mjt_accl(t):
    """Return MJT acceleration trajectory.
    """
    np.clip(t, 0, 1.0, out=t)
    return 60 * np.power(t, 1) - 180 * np.power(t, 2) + 120 * np.power(t, 3)


def get_mjt_pos(xi, xf, T, dt):
    """Returns the MJT position kinematics for the given
    start and end positions, and duration."""
    t = np.arange(0, T + dt, dt) / T
    X = np.array([(xf[i] - xi[i]) * mjt(t) + xi[i]
                  for i in xrange(len(xf))])
    return t, X


def get_mjt_accl(xi, xf, T, dt):
    """Returns the MJT acceleration kinematics for the given
    start and end positions, and duration."""
    t = np.arange(0, T+dt, dt) / T
    X = np.array([(xf[i] - xi[i]) * mjt_accl(t) / np.power(T, 2)
                  for i in xrange(len(xf))])
    return t, X


def get_joint_angles(pos, l1, l2):
    N = np.shape(pos)[1]
    tas = np.zeros((3, N))
    for i in xrange(N):
        tas[:, i] = harm_inv_kinematics(pos[:, i], l1, l2)
    return tas


def get_accl_sensor_signal(accl, grav, arm_dh, tas):
    """Returns the accelerometer signal for the given arm
    movement.
    """
    N = np.shape(accl)[1]
    data = np.zeros((N, 12))
    for i in xrange(N):
        _H, _ = myrob.forward_kinematics(dhparam=arm_dh, t=tas[:, i])
        _R = _H[-1][:3, :3]
        # acceleration and gravity in sensor coordinates
        accl_s = np.matmul(_R.T, np.array([accl[:, i]]).T).T[0]
        grav_s = np.matmul(_R.T, grav).T[0]

        # Arrange data
        data[i, 0:3] = accl[:, i]
        data[i, 3:6] = accl_s
        data[i, 6:9] = grav_s
        data[i, 9:12] = accl_s + grav_s
    return data


def get_gyro_sensor_signal(ep, dt):
    """Returns the gyroscope signal using the given Euler parameters.
    """
    N = np.shape(ep)[0]
    data = np.zeros((N, 6))

    # Angle-axis parameter
    anax = np.array([myrob.euleraram_to_angaxis(_ep) for _ep in ep])
    dphi = (1 / dt) * np.hstack(([0], np.diff(anax[:, 0])))
    # Angular velocity in global coordinates
    w_o = [_dp * _a for _dp, _a in zip(dphi, anax[:, 1:4])]

    # Organize gyro data.
    for i in xrange(N):
        # Rotation matrix.
        _R = myrob.eulerparam_to_rotmat(ep[i, :])

        # Angular velocity in sensor coodinates
        w_s = np.matmul(_R.T, np.array([w_o[i]]).T).T[0]
        # Arrange data
        data[i, 0:3] = w_o[i]
        data[i, 3:6] = w_s

    return data


def organize_data(t, pos, accl, grav, arm_dh, tas, dt=0.001):
    N = len(t)
    data = np.zeros((N, 29))

    # Get Euler parameters for the given movement
    ep = get_eulerparam(arm_dh, tas)

    # Time, position, angles, Euler parameters
    data[:, 0] = t
    data[:, 1:4] = pos.T
    data[:, 4:7] = tas.T
    data[:, 7:11] = ep

    # Get acceleration sensor data
    _data = get_accl_sensor_signal(accl, grav, arm_dh, tas)
    data[:, 11:23] = _data

    # Get gyroscope sensor data
    _data = get_gyro_sensor_signal(ep, dt)
    # Arrange data
    data[:, 23:29] = _data
    return data


def get_eulerparam(arm_dh, tas):
    N = np.shape(tas)[1]
    ep = np.zeros((N, 4))
    for i in xrange(N):
        _H, _ = myrob.forward_kinematics(arm_dh, tas[:, i])
        _Ra = _H[-1][0:3, 0:3]
        ep[i, :] = myrob.rotmat_to_eulerparam(_Ra)
    return ep
