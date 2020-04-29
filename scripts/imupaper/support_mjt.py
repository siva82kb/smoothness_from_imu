"""Supporting routine for generating discrete 3D reaching movements through
convex optimization.

Author: Sivakumar Balasubramanian
Date: 28 Jan 2020
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import toeplitz
from scipy.linalg import block_diag
from cvxopt import matrix, sparse, solvers
from cvxopt.modeling import variable, op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cvxpy as cvx


def minimum_effort_control_2D(D_jerk, Aeq, beq, N):
    '''
    Minimum effort control problem
    minimize    max{||D_jerk * x||}
    subject to  A_eq * x == b_eq
    '''
    # create matrix data type for cvxopt
    # we multiply D_jerk with 1/float_(np.power(N, 3)) to ensure numerical
    # stability
    D_sparse = sparse([matrix(1 / np.float_(np.power(N, 3)) * D_jerk)])
    A_eq = sparse([matrix(Aeq)])
    b_eq = matrix(beq)

    t = variable()  # auxiliary variable
    x = variable(2 * N)  # x, y position of particle
    solvers.options['feastol'] = 1e-6
    solvers.options['show_progress'] = False
    # Linear program
    op(t, [-t <= D_sparse * x, D_sparse * x <= t, A_eq * x == b_eq]).solve()
    return x


def minimum_effort_control_3D(D_jerk, Aeq, beq, N):
    '''
    Minimum effort control problem
    minimize    max{||D_jerk * x||}
    subject to  A_eq * x == b_eq
    '''
    # create matrix data type for cvxopt
    # we multiply D_jerk with 1/float_(power(2 * N, 3)) to ensure numerical
    # stability
    D_sparse = sparse([matrix(1 / np.float_(np.power(N, 3)) * D_jerk)])
    A_eq = sparse([matrix(Aeq)])
    b_eq = matrix(beq)

    t = variable()  # auxiliary variable
    x = variable(3 * N)  # x, y position of particles
    solvers.options['feastol'] = 1e-6
    solvers.options['show_progress'] = False
    # linear program
    op(t, [-t <= D_sparse * x, D_sparse * x <= t, A_eq * x == b_eq]).solve()
    return x


def minimum_effort_control_2D_CVXPY(D_jerk, Aeq, beq, N):
    '''
    Minimum effort control problem
    minimize    max{||D_jerk * x||}
    subject to  A_eq * x == b_eq
    '''
    # Construct the problem.
    x = cvx.Variable((2 * N, 1))
    objective = cvx.Minimize(cvx.sum_squares(D_jerk * x))
    constraints = [Aeq * x == beq]
    prob = cvx.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve(solver=cvx.CVXOPT, verbose=False)

    return prob, x


def minimum_effort_control_3D_CVXPY(D_jerk, Aeq, beq, N):
    '''
    Minimum effort control problem
    minimize    max{||D_jerk * x||}
    subject to  A_eq * x == b_eq
    '''
    # Construct the problem.
    x = cvx.Variable((3 * N, 1))
    objective = cvx.Minimize(cvx.sum_squares(D_jerk * x))
    constraints = [Aeq * x == beq]
    prob = cvx.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve(solver=cvx.CVXOPT, verbose=False)

    return prob, x


def Djerk_matrix(N, dim='2D'):
    """Generates and return the jerk matrix for different dimensions.
    """
    # Define the jerk matrix
    row_jerk = np.hstack((np.array([[-1, 3, -3, 1]]), np.zeros((1, N - 4))))
    col_jerk = np.vstack((np.array([[-1]]), np.zeros((N - 4, 1))))
    _D_jerk = np.power(1, 3) * toeplitz(col_jerk, row_jerk)
    # Check the dimension
    if dim == '1D':
        return _D_jerk
    elif dim == '2D':
        return block_diag(_D_jerk, _D_jerk)
    else:
        return block_diag(_D_jerk, _D_jerk, _D_jerk)


def Aeq_matrix(N, via_pt_times=[], dim='2D'):
    """Generates and returns the Aeq matrix for different dimensions.
    The current version of the matrix only supports via points with
    position specification. Velocity and acceleration costraints are
    not allowed.
    """
    # Initial & final velocity
    init_vel = np.hstack((np.array([[-1, 1]]), np.zeros((1, N - 2))))
    fin_vel = np.hstack((np.zeros((1, N - 2)), np.array([[-1, 1]])))
    # Initial & final acceleration
    init_acc = np.hstack((np.array([[1, -2, 1]]), np.zeros((1, N - 3))))
    fin_acc = np.hstack((np.zeros((1, N - 3)), np.array([[1, -2, 1]])))

    # Not update via point details.
    via_pos = np.zeros((0, N))
    if len(via_pt_times) != 0:
        via_pos = np.zeros((len(via_pt_times), N))
        for i, pt in enumerate(via_pt_times):
            via_pos[i, min(int(pt * N), N - 1)] = 1

    _A = np.vstack((via_pos,
                    init_vel, fin_vel,
                    init_acc, fin_acc))

    # Check the dimension
    if dim == '1D':
        return _A
    elif dim == '2D':
        return block_diag(_A, _A)
    else:
        return block_diag(_A, _A, _A)


def beq_column(via_points, dim='2D'):
    """Generates the b column of the constraint equation.
    """
    _dimn = {'1D': 1, '2D': 2, '3D': 3}
    # make sure via points of the correct size.
    if _dimn[dim] != np.shape(via_points)[1] - 1:
        raise Exception('Via points data dimension is not correct.')
        return

    # Generate the beq for each dimension.
    N_dim = (4 + len(via_points))
    N = _dimn[dim] * N_dim
    beq = np.zeros((N, 1))
    for d in range(_dimn[dim]):
        beq[N_dim * d + 0] = via_points[0, d + 1]  # Initial point
        beq[N_dim * d + 1] = via_points[1, d + 1]  # Final point
        # Now the via points
        for i in range(len(via_points) - 2):
            beq[N_dim * d + i + 2] = via_points[i + 2, d + 1]
    return beq


def plot_trajectory(x_straight, via_points, N, n):
    """Plots the resulting trajectory from an opimtimzation
    problem."""
    # Extract position
    x, y = x_straight.value[0:N], x_straight.value[N:2 * N]

    # Derive velocity
    _row = np.hstack((np.array([[-1, 1]]), np.zeros((1, N - 2))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 2, 1))))
    D_vel = N * toeplitz(_col, _row)
    vx = np.dot(D_vel, x)
    vy = np.dot(D_vel, y)

    # Derive acceleration
    _row = np.hstack((np.array([[1, -2, 1]]), np.zeros((1, N - 3))))
    _col = np.vstack((np.array([[1]]), np.zeros((N - 3, 1))))
    D_acc = np.power(N, 2) * toeplitz(_col, _row)
    ax = np.dot(D_acc, x)
    ay = np.dot(D_acc, y)

    # Derive jerk
    _row = np.hstack((np.array([[-1, 3, -3, 1]]), np.zeros((1, N - 4))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 4, 1))))
    D_jer = np.power(N, 3) * toeplitz(_col, _row)
    jx = np.dot(D_jer, x)
    jy = np.dot(D_jer, y)

    plt.figure(figsize=(15, 5))
    plt.subplot2grid((2, 3), (0, 0), rowspan=3, colspan=1)
    plt.plot(x, y)
    # plot the start/stop and via points
    plt.plot(via_points[0, 1], via_points[0, 2], 'ko', markersize=10)
    plt.plot(via_points[1, 1], via_points[1, 2], 'ks', markersize=10)
    for i in range(len(via_points) - 2):
        plt.plot(via_points[i + 2, 1], via_points[i + 2, 2],
                 'r*', markersize=15)
    plt.ylim(-0.2, 1.8)
    plt.xlim(-1.0, 1.0)
    # axis('equal')

    plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1)
    plt.plot(n, x)
    plt.plot(n, y)
    plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
    plt.plot(n[0:N - 1], vx)
    plt.plot(n[0:N - 1], vy)
    plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1)
    plt.plot(n[0:N - 2], ax)
    plt.plot(n[0:N - 2], ay)
    plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1)
    plt.plot(n[0:N - 3], jx)
    plt.plot(n[0:N - 3], jy)

    plt.tight_layout()


def generate_via_points_2D(Nvia=-1):
    """Randomly generates via points.
    """
    Nvia = np.random.randint(0, 10) if Nvia == -1 else Nvia
    via_points = np.zeros((Nvia + 2, 3))

    # Randomly generate x and y position of the via points.
    x_lims = [-0.25, 0.25]
    y_lims = [0, 1.25]

    via_x = (np.random.rand(Nvia) - 0.5) * (x_lims[1] - x_lims[0])
    via_y = np.random.rand(Nvia) * (y_lims[1] - y_lims[0])
    via_y.sort()

    pts = np.hstack((np.array([[0.], [0.]]),
                     np.vstack((via_x, via_y)),
                     np.array([[0.], [1.]])))

    # Distances between points
    lens = np.array([np.linalg.norm(pt)
                     for pt in (pts[:, 1:] - pts[:, :-1]).T])
    lsum = np.sum(lens)

    # Initial and final points
    via_points[1, :] = [1.0, 0.0, 1.0]
    for i in range(Nvia):
        ti = np.sum(lens[:i + 1]) / lsum
        via_points[i + 2, :] = [ti, pts[0, i + 1], pts[1, i + 1]]

    return via_points


def generate_via_points_3D(Nvia=-1):
    """Randomly generates via points for 3D movements.
    """
    Nvia = np.random.randint(0, 10) if Nvia == -1 else Nvia
    via_points = np.zeros((Nvia + 2, 4))

    # Randomly generate x and y position of the via points.
    x_lims = [-0.25, 0.25]
    y_lims = [0, 1.25]
    z_lims = [-0.25, 0.25]

    via_x = (np.random.rand(Nvia) - 0.5) * (x_lims[1] - x_lims[0])
    via_y = np.random.rand(Nvia) * (y_lims[1] - y_lims[0])
    via_y.sort()
    via_z = (np.random.rand(Nvia) - 0.5) * (z_lims[1] - z_lims[0])

    pts = np.hstack((np.array([[0.], [0.], [0.]]),
                     np.vstack((via_x, via_y, via_z)),
                     np.array([[0.], [1.], [0.]])))

    # Distances between points
    lens = np.array([np.linalg.norm(pt)
                     for pt in (pts[:, 1:] - pts[:, :-1]).T])
    lsum = np.sum(lens)

    # Initial and final points
    via_points[1, :] = [1.0, 0.0, 1.0, 0.0]
    for i in range(Nvia):
        ti = np.sum(lens[:i + 1]) / lsum
        via_points[i + 2, :] = [ti,
                                pts[0, i + 1],
                                pts[1, i + 1],
                                pts[2, i + 1]]
    return via_points


def plot_trajectory_2D(kindata, via_points, N, t):
    """Plots the resulting trajectory from an opimtimzation
    problem."""
    plt.figure(figsize=(15, 5))
    plt.subplot2grid((2, 3), (0, 0), rowspan=3, colspan=1)
    plt.plot(kindata['x'], kindata['y'])
    # plot the start/stop and via points
    plt.plot(via_points[0, 1], via_points[0, 2], 'ko', markersize=10)
    plt.plot(via_points[1, 1], via_points[1, 2], 'ks', markersize=10)
    for i in range(len(via_points) - 2):
        plt.plot(via_points[i + 2, 1], via_points[i + 2, 2],
                 'r*', markersize=15)
    plt.ylim(-0.2, 1.8)
    plt.xlim(-1.0, 1.0)
    plt.axis('equal')

    plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1)
    plt.plot(t, kindata['x'])
    plt.plot(t, kindata['y'])
    plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
    plt.plot(t, kindata['vx'])
    plt.plot(t, kindata['vy'])
    plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1)
    plt.plot(t, kindata['ax'])
    plt.plot(t, kindata['ay'])
    plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1)
    plt.plot(t, kindata['jx'])
    plt.plot(t, kindata['jy'])

    plt.tight_layout()


def plot_trajectory_3D(kindata, via_points, N, n, fname=None):
    """Plots the resulting trajectory from an opimtimzation
    problem."""
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1,
                          projection='3d')
    _off = {'x': -0.75 * np.ones(len(kindata['x'])),
            'y': 1.8 * np.ones(len(kindata['x'])),
            'z': -0.75 * np.ones(len(kindata['x']))}
    plt.plot(kindata['x'], kindata['y'], kindata['z'])
    plt.plot(_off['z'], kindata['y'], kindata['z'], '0.5')
    plt.plot(kindata['x'], _off['y'], kindata['z'], '0.5')
    plt.plot(kindata['x'], kindata['y'], _off['z'], '0.5')
    # plot the start position
    plt.plot([via_points[0, 1]], [via_points[0, 2]], [via_points[0, 3]],
             'k.', markersize=10)
    plt.plot([_off['x'][0]], [via_points[0, 2]], [via_points[0, 3]],
             color='0.3', marker='.', markersize=10)
    plt.plot([via_points[0, 1]], [_off['y'][0]], [via_points[0, 3]],
             color='0.3', marker='.', markersize=10)
    plt.plot([via_points[0, 1]], [via_points[0, 2]], [_off['z'][0]],
             color='0.3', marker='.', markersize=10)
    # plot the stop position
    plt.plot([via_points[1, 1]], [via_points[1, 2]], [via_points[1, 3]],
             'k*', markersize=10)
    plt.plot([_off['x'][0]], [via_points[1, 2]], [via_points[1, 3]],
             color='0.3', marker='*', markersize=10)
    plt.plot([via_points[1, 1]], [_off['y'][0]], [via_points[1, 3]],
             color='0.3', marker='*', markersize=10)
    plt.plot([via_points[1, 1]], [via_points[1, 2]], [_off['z'][0]],
             color='0.3', marker='*', markersize=10)
    # plot the via position(s)
    for i in range(len(via_points) - 2):
        plt.plot([via_points[i + 2, 1]], [via_points[i + 2, 2]],
                 [via_points[i + 2, 3]], 'r.', markersize=10)
        plt.plot([_off['x'][0]], [via_points[i + 2, 2]],
                 [via_points[i + 2, 3]], color=[1.0, 0.5, 0.5],
                 marker='.', markersize=10)
        plt.plot([via_points[i + 2, 1]], [_off['y'][0]],
                 [via_points[i + 2, 3]], color=[1.0, 0.5, 0.5],
                 marker='.', markersize=10)
        plt.plot([via_points[i + 2, 1]], [via_points[i + 2, 2]],
                 [_off['z'][0]], color=[1.0, 0.5, 0.5],
                 marker='.', markersize=10)
    ax.set_ylim(-0.2, 1.8)
    ax.set_xlim(-0.75, 0.75)
    ax.set_zlim(-0.75, 0.75)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.zaxis.set_tick_params(labelsize=10)


    plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1)
    plt.plot(n, kindata['x'])
    plt.plot(n, kindata['y'])
    plt.plot(n, kindata['z'])
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Position', fontsize=15)
    plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
    plt.plot(n, kindata['vx'])
    plt.plot(n, kindata['vy'])
    plt.plot(n, kindata['vz'])
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Velocity', fontsize=15)
    plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1)
    plt.plot(n, kindata['ax'])
    plt.plot(n, kindata['ay'])
    plt.plot(n, kindata['az'])
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Acceleration', fontsize=15)
    plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1)
    plt.plot(n, kindata['jx'])
    plt.plot(n, kindata['jy'])
    plt.plot(n, kindata['jz'])
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Jerk', fontsize=15)

    plt.tight_layout()
    if fname is not None:
        fig.savefig("{0}.svg".format(fname), dpi=300, format="svg")
        plt.close(fig)


def extract_all_kinematics_2D(move, N, dt):
    """Extracts position, velocity, acceleration and jerk from the optimal
    solution."""

    # Extract position
    x, y = (move.value[0:N],
            move.value[N:2 * N])

    # Derive velocity
    _row = np.hstack((np.array([[-1, 1]]), np.zeros((1, N - 2))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 2, 1))))
    D_vel = toeplitz(_col, _row) / dt
    vx = np.vstack((np.dot(D_vel, x), np.array([[0.]])))
    vy = np.vstack((np.dot(D_vel, y), np.array([[0.]])))

    # Derive acceleration
    _row = np.hstack((np.array([[1, -2, 1]]), np.zeros((1, N - 3))))
    _col = np.vstack((np.array([[1]]), np.zeros((N - 3, 1))))
    D_acc = toeplitz(_col, _row) / np.power(dt, 2)
    ax = np.vstack((np.dot(D_acc, x), np.array([[0.], [0.]])))
    ay = np.vstack((np.dot(D_acc, y), np.array([[0.], [0.]])))

    # Derive jerk
    _row = np.hstack((np.array([[-1, 3, -3, 1]]), np.zeros((1, N - 4))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 4, 1))))
    D_jer = toeplitz(_col, _row) / np.power(dt, 3)
    jx = np.vstack((np.dot(D_jer, x), np.array([[0.], [0.], [0.]])))
    jy = np.vstack((np.dot(D_jer, y), np.array([[0.], [0.], [0.]])))

    return pd.DataFrame.from_dict({'x': list(x),
                                   'y': list(y),
                                   'vx': list(vx.T)[0],
                                   'vy': list(vy.T)[0],
                                   'ax': list(ax.T)[0],
                                   'ay': list(ay.T)[0],
                                   'jx': list(jx.T)[0],
                                   'jy': list(jy.T)[0]})


def extract_all_kinematics_2D_CVXPY(move, N, dt):
    """Extracts position, velocity, acceleration and jerk from the optimal
    solution."""

    # Extract position
    x, y = (move.value[0:N],
            move.value[N:2 * N])

    # Derive velocity
    _row = np.hstack((np.array([[-1, 1]]), np.zeros((1, N - 2))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 2, 1))))
    D_vel = toeplitz(_col, _row) / dt
    vx = np.vstack((np.dot(D_vel, x), np.array([[0.]])))
    vy = np.vstack((np.dot(D_vel, y), np.array([[0.]])))

    # Derive acceleration
    _row = np.hstack((np.array([[1, -2, 1]]), np.zeros((1, N - 3))))
    _col = np.vstack((np.array([[1]]), np.zeros((N - 3, 1))))
    D_acc = toeplitz(_col, _row) / np.power(dt, 2)
    ax = np.vstack((np.dot(D_acc, x), np.array([[0.], [0.]])))
    ay = np.vstack((np.dot(D_acc, y), np.array([[0.], [0.]])))

    # Derive jerk
    _row = np.hstack((np.array([[-1, 3, -3, 1]]), np.zeros((1, N - 4))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 4, 1))))
    D_jer = toeplitz(_col, _row) / np.power(dt, 3)
    jx = np.vstack((np.dot(D_jer, x), np.array([[0.], [0.], [0.]])))
    jy = np.vstack((np.dot(D_jer, y), np.array([[0.], [0.], [0.]])))

    return pd.DataFrame.from_dict({'x': x.T.tolist()[0],
                                   'y': y.T.tolist()[0],
                                   'vx': vx.T.tolist()[0],
                                   'vy': vy.T.tolist()[0],
                                   'ax': ax.T.tolist()[0],
                                   'ay': ay.T.tolist()[0],
                                   'jx': jx.T.tolist()[0],
                                   'jy': jy.T.tolist()[0]})


def extract_all_kinematics_3D(move, N, dt):
    """Extracts position, velocity, acceleration and jerk from the optimal
    solution."""

    # Extract position
    x, y, z = (move.value[0:N],
               move.value[N:2 * N],
               move.value[2 * N:3 * N])

    # Derive velocity
    _row = np.hstack((np.array([[-1, 1]]), np.zeros((1, N - 2))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 2, 1))))
    D_vel = N * toeplitz(_col, _row)
    vx = np.vstack((np.dot(D_vel, x), np.array([[0.]])))
    vy = np.vstack((np.dot(D_vel, y), np.array([[0.]])))
    vz = np.vstack((np.dot(D_vel, z), np.array([[0.]])))

    # Derive acceleration
    _row = np.hstack((np.array([[1, -2, 1]]), np.zeros((1, N - 3))))
    _col = np.vstack((np.array([[1]]), np.zeros((N - 3, 1))))
    D_acc = np.power(N, 2) * toeplitz(_col, _row)
    ax = np.vstack((np.dot(D_acc, x), np.array([[0.], [0.]])))
    ay = np.vstack((np.dot(D_acc, y), np.array([[0.], [0.]])))
    az = np.vstack((np.dot(D_acc, z), np.array([[0.], [0.]])))

    # Derive jerk
    _row = np.hstack((np.array([[-1, 3, -3, 1]]), np.zeros((1, N - 4))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 4, 1))))
    D_jer = np.power(N, 3) * toeplitz(_col, _row)
    jx = np.vstack((np.dot(D_jer, x), np.array([[0.], [0.], [0.]])))
    jy = np.vstack((np.dot(D_jer, y), np.array([[0.], [0.], [0.]])))
    jz = np.vstack((np.dot(D_jer, z), np.array([[0.], [0.], [0.]])))

    return pd.DataFrame.from_dict({'x': list(x),
                                   'y': list(y),
                                   'z': list(z.T),
                                   'vx': list(vx.T)[0],
                                   'vy': list(vy.T)[0],
                                   'vz': list(vz.T)[0],
                                   'ax': list(ax.T)[0],
                                   'ay': list(ay.T)[0],
                                   'az': list(az.T)[0],
                                   'jx': list(jx.T)[0],
                                   'jy': list(jy.T)[0],
                                   'jz': list(jz.T)[0]})


def extract_all_kinematics_3D_CVXPY(move, N, dt):
    """Extracts position, velocity, acceleration and jerk from the optimal
    solution."""

    # Extract position
    x, y, z = (move.value[0:N],
               move.value[N:2 * N],
               move.value[2 * N:3 * N])

    # Derive velocity
    _row = np.hstack((np.array([[-1, 1]]), np.zeros((1, N - 2))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 2, 1))))
    D_vel = toeplitz(_col, _row) / dt
    vx = np.vstack((np.dot(D_vel, x), np.array([[0.]])))
    vy = np.vstack((np.dot(D_vel, y), np.array([[0.]])))
    vz = np.vstack((np.dot(D_vel, z), np.array([[0.]])))

    # Derive acceleration
    _row = np.hstack((np.array([[1, -2, 1]]), np.zeros((1, N - 3))))
    _col = np.vstack((np.array([[1]]), np.zeros((N - 3, 1))))
    D_acc = toeplitz(_col, _row) / np.power(dt, 2)
    ax = np.vstack((np.dot(D_acc, x), np.array([[0.], [0.]])))
    ay = np.vstack((np.dot(D_acc, y), np.array([[0.], [0.]])))
    az = np.vstack((np.dot(D_acc, z), np.array([[0.], [0.]])))

    # Derive jerk
    _row = np.hstack((np.array([[-1, 3, -3, 1]]), np.zeros((1, N - 4))))
    _col = np.vstack((np.array([[-1]]), np.zeros((N - 4, 1))))
    D_jer = toeplitz(_col, _row) / np.power(N, 3)
    jx = np.vstack((np.dot(D_jer, x), np.array([[0.], [0.], [0.]])))
    jy = np.vstack((np.dot(D_jer, y), np.array([[0.], [0.], [0.]])))
    jz = np.vstack((np.dot(D_jer, z), np.array([[0.], [0.], [0.]])))

    return pd.DataFrame.from_dict({'x': x.T.tolist()[0],
                                   'y': y.T.tolist()[0],
                                   'z': z.T.tolist()[0],
                                   'vx': vx.T.tolist()[0],
                                   'vy': vy.T.tolist()[0],
                                   'vz': vz.T.tolist()[0],
                                   'ax': ax.T.tolist()[0],
                                   'ay': ay.T.tolist()[0],
                                   'az': az.T.tolist()[0],
                                   'jx': jx.T.tolist()[0],
                                   'jy': jy.T.tolist()[0],
                                   'jz': jz.T.tolist()[0]})


def generate_movements_3D(via_points, N, dim):
    """Generates the movement for the given set of via points.
    """
    D_jerk = Djerk_matrix(N, dim)
    A = Aeq_matrix(N, via_pt_times=via_points[:, 0], dim=dim)
    b = beq_column(via_points, dim=dim)

    # Solve.
    return minimum_effort_control_3D_CVXPY(D_jerk, A, b, N)
