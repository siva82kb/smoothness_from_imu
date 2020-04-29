"""Module containing the necessary variables and functions for generating and
analysing smoothness from the virtual IMU during simple reaching movements.
"""

import numpy as np
import json
import sys
import glob
import matplotlib.pyplot as plt

# sys.path.append("../scripts/")
import imupaper.virtualimu as vimu
from smoothness import sparc, log_dimensionless_jerk


class Params():
    # Defining the variables that will be used in the analysis
    # 3DOF Human Arm Model
    # Distnaces are in centimeters.
    L1, L2 = 30.0, 32.0
    arm_dh = [{'a': 0, 'al': np.pi/2, 'd': 0, 't': 0},
              {'a': L1, 'al': 0, 'd': 0, 't': 0},
              {'a': L2, 'al': -np.pi/2, 'd': 0, 't': 0}]

    # Target information
    O = 30 * np.array([[1 / np.sqrt(2), 0, 0]])
    pts = np.array([[+0.5, +0.5, +0.5],  # Front face
                    [+0.5, -0.5, +0.5],
                    [+0.5, -0.5, -0.5],
                    [+0.5, +0.5, -0.5],
                    [-0.5, +0.5, +0.5],  # Back face
                    [-0.5, -0.5, +0.5],
                    [-0.5, -0.5, -0.5],
                    [-0.5, +0.5, -0.5],
                    [+0.5, +0.0, +0.0],  # Front center
                    [-0.5, +0.0, +0.0],  # Back center
                    [+0.0, +0.5, +0.0],  # Left center
                    [+0.0, -0.5, +0.0],  # Right center
                    [+0.0, +0.0, +0.5],  # Top center
                    [+0.0, +0.0, -0.5],  # Bottom center
                    ])

    # Movement amplitude and duration
    Amp = 30
    Dur = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    Tgts = Amp * pts + O
    dt = 0.001
    fs = 1 / dt
    Ndur, Ntgt = len(Dur), len(Tgts)

    # gravity
    grav = np.array([[0, 0, -98.1]]).T

    # out directory
    outdir = "../virtualimu_data/simple"

    # Data header.
    headcols = ("time",
                "px", "py", "pz",
                "ta1", "ta2", "ta3",
                "ep1", "ep2", "ep3", "ep4",
                "ax", "ay", "az",
                "axs", "ays", "azs",
                "gxs", "gys", "gzs",
                "agxs", "agys", "agzs",
                "gyx", "gyy", "gyz",
                "gyxs", "gyys", "gyzs")
    header = ", ".join(headcols)

    # Indices for the different variables of interest
    posInx = [1, 2, 3]
    acclInx = [11, 12, 13]
    acclsInx = [20, 21, 22]

    @staticmethod
    def write_params_file():
        # Details of the simulated movements
        params = {'arm': {'l1': Params.L1,
                            'l2': Params.L2,
                            'dh': Params.arm_dh},
                    'origin': Params.O.tolist(),
                    'gravity': Params.grav.tolist(),
                    'pts': Params.pts.tolist(),
                    'amp': Params.Amp, 'dur': Params.Dur,
                    'tgts': Params.Tgts.tolist(),
                    'dt': Params.dt}
        # Write the parameter file
        # Write params file
        with open("{0}/params.json".format(Params.outdir), "w") as fh:
            json.dump(params, fh, indent=4)


def target_time_combs(tgts, T):
    for i, _T in enumerate(T):
        for j, _tgt in enumerate(tgts):
            yield i, j, _T, _tgt


def generate_save_simple_movements(params):
    """Generates movement data and saves files to the disk.
    """
    allcombs = target_time_combs(params.Tgts, params.Dur)
    for i, j, _t, _tgt in allcombs:
        # Get movement trajectory
        t, pos = vimu.get_mjt_pos(params.O[0], _tgt, _t, params.dt)
        t, accl = vimu.get_mjt_accl(params.O[0], _tgt, _t, params.dt)

        # Joint angles, end-point orientation and gravity component
        tas = vimu.get_joint_angles(pos, params.L1, params.L2)

        # organize all data
        data = vimu.organize_data(t, pos, accl, params.grav, params.arm_dh,
                                  tas, params.dt)

        # Save data to files.
        _fname = "{0}/data_{1}_{2}.csv".format(params.outdir, i, j)
        _str = "\rWriting Dur: {0:0.2f}s Tgt:{1:3d} | {2}"
        sys.stdout.write(_str.format(params.Dur[i], j, _fname))
        np.savetxt(_fname, data, delimiter=", ", fmt="%10.10f",
                   header=params.header)

    sys.stdout.write("\nDone!")


def analyse_smoothness(params):
    # Initialize smoothness values variable.
    smoothvals = {'sparc': np.zeros((params.Ndur, params.Ntgt)),
                  'ldljv': np.zeros((params.Ndur, params.Ntgt)),
                  'ldlja': np.zeros((params.Ndur, params.Ntgt)),
                  'sparcs': np.zeros((params.Ndur, params.Ntgt)),
                  'ldljsv': np.zeros((params.Ndur, params.Ntgt)),
                  'ldljsa': np.zeros((params.Ndur, params.Ntgt)),
                  'sparcs-wom': np.zeros((params.Ndur, params.Ntgt)),
                  'ldljsv-wom': np.zeros((params.Ndur, params.Ntgt)),
                  'ldljsa-wom': np.zeros((params.Ndur, params.Ntgt)),
                  'agr': np.zeros((params.Ndur, params.Ntgt)),
                  }
    files = glob.glob("{0}/data_*.csv".format(params.outdir))

    for i, f in enumerate(files):
        dinx, tinx = get_dur_tgt_index_from_fname(f)

        # Read accl, vel and speed data
        mdata = read_get_vel_accl_data(f, params)

        # Remove mean for each accelerometer sensor data
        mdata['accls-wom'] = mdata['accls'] - np.mean(mdata['accls'], axis=0)
        mdata['vels-wom'] = np.cumsum(mdata['accls-wom'], axis=0) * params.dt
        mdata['spds-wom'] = np.linalg.norm(mdata['vels-wom'], axis=1)

        # Estimate AGR.
        _g = (np.sqrt(len(mdata['accl'].T)) * 98.1)
        agr = 20 * np.log10(np.linalg.norm(mdata['accl']) / _g)

        # Calculate smoothness - SPARC
        _sparc, _, _ = sparc(mdata['spd'], fs=params.fs)
        _sparcs, _, _ = sparc(mdata['spds'], fs=params.fs)
        _sparcswom, _, _ = sparc(mdata['spds-wom'], fs=params.fs)

        # Calculate smoothness - LDLJ
        # From velocity
        _ldljv = log_dimensionless_jerk(mdata['vel'], fs=params.fs,
                                        data_type="vel")
        _ldljsv = log_dimensionless_jerk(mdata['vels'], fs=params.fs,
                                         data_type="vel")
        _ldljsvwom = log_dimensionless_jerk(mdata['vels-wom'], fs=params.fs,
                                            data_type="vel")
        # From velocity
        _ldlja = log_dimensionless_jerk(mdata['accl'], fs=params.fs,
                                        data_type="accl")
        _ldljsa = log_dimensionless_jerk(mdata['accls'], fs=params.fs,
                                         data_type="accl")
        _ldljsawom = log_dimensionless_jerk(mdata['accls-wom'], fs=params.fs,
                                            data_type="accl")

        # Save data.
        smoothvals['sparc'][dinx, tinx] = _sparc
        smoothvals['sparcs'][dinx, tinx] = _sparcs
        smoothvals['sparcs-wom'][dinx, tinx] = _sparcswom
        smoothvals['ldljv'][dinx, tinx] = _ldljv
        smoothvals['ldljsv'][dinx, tinx] = _ldljsv
        smoothvals['ldljsv-wom'][dinx, tinx] = _ldljsvwom
        smoothvals['ldlja'][dinx, tinx] = _ldlja
        smoothvals['ldljsa'][dinx, tinx] = _ldljsa
        smoothvals['ldljsa-wom'][dinx, tinx] = _ldljsawom
        smoothvals['agr'][dinx, tinx] = agr

        sys.stdout.write("\r {0:3d}/{1:3d} | {2}".format(i, len(files), f))
    sys.stdout.write("\nDone!")

    return smoothvals


def get_dur_tgt_index_from_fname(f):
    # Get file details
    _v = f.split('/')[-1].split('.')[0].split('_')
    return int(_v[1]), int(_v[2])


def read_get_vel_accl_data(f, params):
    # Read data file.
    data = np.loadtxt(fname=f, delimiter=', ')

    mdata = {}
    # Position data
    mdata['pos'] = data[:, params.posInx]

    # Calculate velocity from acceleration
    mdata['accl'] = data[:, params.acclInx]
    mdata['vel'] = np.cumsum(mdata['accl'], axis=0) * params.dt
    mdata['spd'] = np.linalg.norm(mdata['vel'], axis=1)

    # Calculate velocity from accelerometer sensor data
    mdata['accls'] = data[:, params.acclsInx]
    mdata['vels'] = np.cumsum(mdata['accls'], axis=0) * params.dt
    mdata['spds'] = np.linalg.norm(mdata['vels'], axis=1)

    return mdata


def generate_smoothness_sgr_summary(smoothvals, params):
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(241)
    plt.boxplot(smoothvals['agr'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    plt.xticks(np.arange(1, len(params.Dur) + 1), params.Dur)
    plt.xlabel("Movement duration (sec)")
    plt.ylabel("Accl-to-Gravity Ratio (dB)")
    plt.title("AGR vs. Movement duration")

    ax = fig.add_subplot(242)
    plt. boxplot(smoothvals['sparc'].T,
                 notch=False,  # notch shape
                 vert=True,  # vertical box alignment
                 patch_artist=True,  # fill with color
                 whis=2.5
                 )
    plt.boxplot(smoothvals['sparcs'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    ax.set_ylim(-2.2, -1.3)
    plt.xticks(np.arange(1, len(params.Dur) + 1), params.Dur)
    plt.xlabel("Movement duration (sec)")
    plt.ylabel("SPARC Smoothness")
    plt.title("SPARC vs. Movement duration")

    ax = fig.add_subplot(246)
    plt. boxplot(smoothvals['sparc'].T,
                 notch=False,  # notch shape
                 vert=True,  # vertical box alignment
                 patch_artist=True,  # fill with color
                 whis=2.5
                 )
    plt.boxplot(smoothvals['sparcs-wom'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    ax.set_ylim(-2.2, -1.3)
    plt.xticks(np.arange(1, len(params.Dur) + 1), params.Dur)
    plt.xlabel("Movement duration (sec)")
    plt.ylabel("SPARC Smoothness")
    plt.title("SPARC (WOM) vs. Movement duration")

    ax = fig.add_subplot(243)
    plt.boxplot(smoothvals['ldljv'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                whis=2.5
                )
    plt.boxplot(smoothvals['ldljsv'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    plt.xticks(np.arange(1, len(params.Dur) + 1), params.Dur)
    ax.set_ylim(-6, 6)
    plt.xlabel("Movement duration (sec)")
    plt.ylabel("LDLJ-V Smoothness")
    plt.title("LDLJ-V vs. Movement duration")

    ax = fig.add_subplot(247)
    plt.boxplot(smoothvals['ldljv'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                whis=2.5
                )
    plt.boxplot(smoothvals['ldljsv-wom'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    plt.xticks(np.arange(1, len(params.Dur) + 1), params.Dur)
    ax.set_ylim(-6, 6)
    plt.xlabel("Movement duration (sec)")
    plt.ylabel("LDLJ-V Smoothness")
    plt.title("LDLJ-V (WOM) vs. Movement duration")

    ax = fig.add_subplot(244)
    plt.boxplot(smoothvals['ldlja'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                whis=2.5
                )
    plt.boxplot(smoothvals['ldljsa'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    plt.xticks(np.arange(1, len(params.Dur) + 1), params.Dur)
    ax.set_ylim(-6, 6)
    plt.xlabel("Movement duration (sec)")
    plt.ylabel("LDLJ-A Smoothness")
    plt.title("LDLJ-A vs. Movement duration")

    ax = fig.add_subplot(248)
    plt.boxplot(smoothvals['ldlja'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                whis=2.5
                )
    plt.boxplot(smoothvals['ldljsa-wom'].T,
                notch=False,  # notch shape
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    plt.xticks(np.arange(1, len(params.Dur) + 1), params.Dur)
    ax.set_ylim(-6, 6)
    plt.xlabel("Movement duration (sec)")
    plt.ylabel("LDLJ-A Smoothness")
    plt.title("LDLJ-A (WOM) vs. Movement duration")

    plt.tight_layout()

    return fig
