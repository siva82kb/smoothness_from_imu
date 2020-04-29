"""Module containing the necessary variables and functions for generating and
analysing smoothness from the virtual IMU during complex reaching movements.
"""

import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd
import seaborn as sb

# sys.path.append("../scripts/")
import myrobotics as myrob
import imupaper.virtualimu as vimu
from smoothness import sparc, log_dimensionless_jerk


def GetParamDetailsForFile(f):
    _nv, _nr = f.split('.csv')[0].split('/')[-1].split('via_')
    _nr = _nr.split('_new')[0]
    return int(_nv), int(_nr)


def GetParamDetailsForFiles(files):
    nvia = []
    nrep = []
    for f in files:
        _split = GetParamDetailsForFile(f)
        nvia.append(int(_split[0]))
        nrep.append(int(_split[1]))
    return list(set(nvia)), list(set(nrep))


class Params():
    # 3DOF Human Arm Model
    L1, L2 = 30.0, 32.0
    arm_dh = [{'a': 0, 'al': np.pi/2, 'd': 0, 't': 0},
              {'a': L1, 'al': 0, 'd': 0, 't': 0},
              {'a': L2, 'al': -np.pi/2, 'd': 0, 't': 0}]

    # Target information
    O = 30 * np.array([[1 / np.sqrt(2), 0, 0]])
    orientations = np.pi * np.array([[-0.25, +0.25],
                                     [-0.75, +0.25],
                                     [-0.75, -0.25],
                                     [-0.25, -0.25],
                                     [+0.25, +0.25],
                                     [+0.75, +0.25],
                                     [+0.75, -0.25],
                                     [+0.25, -0.25],
                                     [-0.50, +0.00],
                                     [+0.50, +0.00],
                                     [+0.00, +0.00],
                                     [+0.00, +1.00],
                                     [+0.00, +0.50],
                                     [+0.00, -0.50]])

    # Movement amplitude and duration
    Amp = 15
    Dur = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dt = 0.001
    fs = 1 / dt
    Ndur = len(Dur)
    Norien = len(orientations)

    # Data related stuff.
    datadir = "../virtualimu_data/data"
    files = glob.glob("{0}/*.csv".format(datadir))
    # Get parameters
    nVia, _ = GetParamDetailsForFiles(files)
    Nrep = 1
    # Rep indices used for analysis
    repInx = [[0]] + np.random.randint(low=0, high=10,
                                       size=(len(nVia), Nrep)).tolist()

    # gravity
    grav = np.array([[0, 0, -98.1]]).T

    # # out directory
    outdir = "../virtualimu_data/complex/"
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
    acclinsInx = [14, 15, 16]
    gravinsInx = [17, 18, 19]
    acclsInx = [20, 21, 22]

    @staticmethod
    def write_params_file():
        # Details of the simulated movements
        params = {'arm': {'l1': Params.L1,
                          'l2': Params.L2,
                          'dh': Params.arm_dh},
                  'origin': Params.O.tolist(),
                  'gravity': Params.grav.tolist(),
                  'amp': Params.Amp, 'dur': Params.Dur,
                  'orientations': Params.orientations.tolist(),
                  'dt': Params.dt,
                  'datadir': Params.datadir,
                  'files': Params.files, 'nVia': Params.nVia,
                  'Nrep': Params.Nrep, 'repInx': Params.repInx}
        # Write the parameter file
        # Write params file
        with open("{0}/params.json".format(Params.outdir), "w") as fh:
            json.dump(params, fh, indent=4)


def get_joint_pos(dhparam, t, jinx):
    H, _ = myrob.forward_kinematics(dhparam, t)
    return H[jinx][:3, 3]


def get_joint_rotmat(dhparam, t, jinx):
    H, _ = myrob.forward_kinematics(dhparam, t)
    return H[jinx][0:3, 0:3]


def get_max_reconstruct_error(dhparam, tas, pos):
    _pos = np.array([get_joint_pos(dhparam, t, 2) for t in tas.T]).T
    return np.linalg.norm(_pos - pos, axis=0)


def get_max_g_error(dhparam, tas, accls, accl, grav):
    _err = np.array([np.matmul(get_joint_rotmat(dhparam, t, 2), accls[:, i]) -
                     accl[:, i] - grav[:, 0] for i, t in enumerate(tas.T)]).T
    return np.linalg.norm(_err, axis=0)


def read_complex_movement_data(params):
    # File name format.
    fname_str = "{0}/{{0:02d}}via_{{1:04d}}.csv".format(params.datadir)

    for nv in params.nVia:
        for j, nr in enumerate(params.repInx[nv]):
            f = fname_str.format(nv, params.repInx[nv][j])
            # Read file.
            data = pd.read_csv(f)
            # Generate movements with different orientations.
            for k, angs in enumerate(params.orientations):
                yield j, k, nv, nr, angs, data, f


def read_get_vel_accl_data(f, dur, params, gravper=100.0):
    # Read data file.
    data = np.loadtxt(fname=f, delimiter=', ')

    mdata = {}
    # Position data
    mdata['pos'] = data[:, params.posInx]

    # Calculate velocity from acceleration
    mdata['accl'] = (1 / np.power(dur, 2)) * data[:, params.acclInx]
    mdata['vel'] = np.cumsum(mdata['accl'], axis=0) * params.dt * dur
    mdata['spd'] = np.linalg.norm(mdata['vel'], axis=1)

    # Acceleration and gravity in sensor coordinates
    mdata['acclins'] = (1 / np.power(dur, 2)) * data[:, params.acclinsInx]
    mdata['gravins'] = (gravper / 100.0) * data[:, params.gravinsInx]

    # Calculate velocity from accelerometer sensor data
    mdata['accls'] = mdata['acclins'] + mdata['gravins']
    mdata['vels'] = np.cumsum(mdata['accls'], axis=0) * params.dt * dur
    mdata['spds'] = np.linalg.norm(mdata['vels'], axis=1)

    # Calculate velocity and acceleration from sensor data without mean.
    mdata['accl-wom'] = mdata['accl'] - np.mean(mdata['accl'], axis=0)
    mdata['vel-wom'] = np.cumsum(mdata['accl-wom'], axis=0) * params.dt * dur
    mdata['spd-wom'] = np.linalg.norm(mdata['vel-wom'], axis=1)

    # Calculate velocity and acceleration from sensor data without mean.
    mdata['accls-wom'] = mdata['accls'] - np.mean(mdata['accls'], axis=0)
    mdata['vels-wom'] = np.cumsum(mdata['accls-wom'], axis=0) * params.dt * dur
    mdata['spds-wom'] = np.linalg.norm(mdata['vels-wom'], axis=1)

    return mdata


def generate_save_complex_movements(params):
    alldata = read_complex_movement_data(params)
    # Virutal IMU data for complex reaching movements with several via points.
    _fnamestr = "{0}/data_{1}_{2}({3})_{4}.csv"
    _dispstr = " | ".join(("\rWriting [{0:02d}/{6:02d}] [{7:02d}/{8:02d}]", 
                           "{0:02d}/{1:02d}/{2:02d}", "{3}",
                           "pErr: {4:0.5f}", "gErr: {5:0.5f}"))
    for i, j, nv, nr, angs, data, f in alldata:
        # Rotate movement to the appropriate orientation.
        _R = np.matmul(myrob.rotz(angs[0]), myrob.rotx(angs[1]))

        # Rotate position and acceleration data to the required orientation.
        pos = np.array(data[['x', 'y', 'z']]).T
        pos = params.Amp * np.matmul(_R, pos) + params.O.T
        accl = np.array(data[['ax', 'ay', 'az'] ]).T
        accl = params.Amp * np.matmul(_R, accl)

        # Inverse kinematics
        tas = vimu.get_joint_angles(pos, params.L1, params.L2)

        # organize all data
        t = np.arange(0, params.dt * len(data), params.dt)
        imudata = vimu.organize_data(t, pos, accl, params.grav, params.arm_dh,
                                     tas, params.dt)

        # Position reconstruction and gravity error.
        err = get_max_reconstruct_error(params.arm_dh, tas, pos)
        err = err.reshape((len(err), 1))
        gerr = get_max_g_error(params.arm_dh, tas, imudata[:, 20:23].T, accl,
                               params.grav)
        gerr = gerr.reshape((len(gerr), 1))

        # Save data
        np.savetxt(_fnamestr.format(params.outdir, nv, i, nr, j),
                   np.hstack((imudata, err, gerr)), delimiter=", ",
                   fmt="%10.10f", header=params.header)

        sys.stdout.write(_dispstr.format(nv, nr, j, f.split('/')[-1],
                                         np.max(err), np.max(gerr),
                                         len(params.nVia), i, params.Nrep))
    sys.stdout.write("\nDone!")


def _smoothsparc(sp, sps, spwom, spswom, fs):
    _ss, _, _ = sparc(sp, fs)
    _sss, _, _ = sparc(sps, fs)
    _sswom, _, _ = sparc(spwom, fs)
    _ssswom, _, _ = sparc(spswom, fs)
    return _ss, _sss, _sswom, _ssswom


def _smoothldljv(v, vs, vwom, vswom, fs):
    _sl = log_dimensionless_jerk(v, fs=fs, data_type="vel")
    _sls = log_dimensionless_jerk(vs, fs=fs, data_type="vel")
    _slwom = log_dimensionless_jerk(vwom, fs=fs, data_type="vel")
    _slswom = log_dimensionless_jerk(vswom, fs=fs, data_type="vel")
    return _sl, _sls, _slwom, _slswom


def _smoothldlja(ac, acs, acwom, acswom, fs):
    _sl = log_dimensionless_jerk(ac, fs=fs, data_type="accl")
    _sls = log_dimensionless_jerk(acs, fs=fs, data_type="accl")
    _slwom = log_dimensionless_jerk(acwom, fs=fs, data_type="accl")
    _slswom = log_dimensionless_jerk(acswom, fs=fs, data_type="accl")
    return _sl, _sls, _slwom, _slswom


def estimate_reconstruction_performance(params, gravper=100):
    # Velocity reconstruction error.
    _sz = (len(Params.nVia), Params.Nrep,
           len(Params.orientations), len(Params.Dur))
    cols = ["via", "rep", "orien", "dur",
            "sgr", "err", "corr",
            "sparc", "sparcs", "sparc-wom", "sparcs-wom",
            "ldljv", "ldljsv", "ldljv-wom", "ldljsv-wom",
            "ldlja", "ldljsa", "ldlja-wom", "ldljsa-wom"]
    velRecon = pd.DataFrame(columns=cols)

    _dispStr = "\r [{3}%] {0:05d}/{1:05d} | {2} \t\t\t"

    all_files = get_all_files_info(Params)
    for nv, nr, no, nd, f in all_files:
        # Read accl, vel and speed data
        mdata = read_get_vel_accl_data(f[1], nd[1], Params, gravper=gravper)

        # Sensor to gravity ratio
        _g = np.sqrt(len(mdata['accls'])) * np.linalg.norm(Params.grav)
        _sgr = np.linalg.norm(mdata['accls']) / _g

        # Speed reconstruction error
        _temp = np.max(np.abs(mdata['spd'] - mdata['spds-wom']))
        _temp1 = np.max(mdata['spd'])
        _err = _temp / _temp1

        # Speed correlation
        _corr = np.corrcoef(np.linalg.norm(mdata['vel'], axis=1),
                            np.linalg.norm(mdata['vels-wom'], axis=1))[0, 1]

        # Smoothness analysis
        # SPARC
        _sparc = _smoothsparc(mdata['spd'], mdata['spds'],
                              mdata['spd-wom'], mdata['spds-wom'],
                              params.fs)
        # LDLJ velocity
        _ldljv = _smoothldljv(mdata['vel'], mdata['vels'],
                              mdata['vel-wom'], mdata['vels-wom'],
                              params.fs)
        # LDLJ acceleration
        _ldlja = _smoothldljv(mdata['accl'], mdata['accls'],
                              mdata['accl-wom'], mdata['accls-wom'],
                              params.fs)

        # Append to dataframe
        _data = {'via': [nv[1]], 'rep': [nr],
                 'orien': [no], 'dur': [nd[1]],
                 'sgr': [_sgr], 'err': [_err], 'corr': [_corr],
                 'sparc': _sparc[0], 'sparcs': _sparc[1],
                 'sparc-wom': _sparc[2], 'sparcs-wom': _sparc[3],
                 'ldljv': _ldljv[0], 'ldljsv': _ldljv[1],
                 'ldljv-wom': _ldljv[2], 'ldljsv-wom': _ldljv[3],
                 'ldlja': _ldlja[0], 'ldljsa': _ldlja[1],
                 'ldlja-wom': _ldlja[2], 'ldljsa-wom': _ldlja[3],
                 }
        velRecon = velRecon.append(pd.DataFrame.from_dict(_data),
                                   ignore_index=True)
        #  velRecon['corr'][nv[0], nr, no, nd[0]] = _temp[0, 1]
        sys.stdout.write(_dispStr.format(f[0], f[2], f[1].split('/')[-1],
                                         gravper))
        sys.stdout.flush()

    return velRecon


def get_all_files_info(params):
    files = glob.glob("{0}/data_*.csv".format(params.outdir))
    for nf, f in enumerate(files):
        # Get details about file.
        _temp = f.split('/')[-1].split('.')[0].split('_')[1:]
        nv = int(_temp[0])
        nr = int(_temp[1].split('(')[0])
        no = int(_temp[2])
        for nd, dur in enumerate(params.Dur):
            _v = params.nVia.index(nv)
            yield (_v, nv), nr, no, (nd, dur), (nf, f, len(files))


def _org_by_dur(velRecon, col):
    durs = velRecon['dur'].unique()
    _dataprctnls = []
    for d in durs:
        _dinx = velRecon['dur'] == d
        _data = velRecon.loc[_dinx, col].dropna()
        _dataprctnls.append([np.percentile(_data, q=q)
                             for q in [25, 50, 75]])
    return np.array(_dataprctnls)


def _org_by_dur_reldata(velRecon, col1, col2):
    durs = velRecon['dur'].unique()
    _dataprctnls = []
    for d in durs:
        _dinx = velRecon['dur'] == d
        _d1 = velRecon[_dinx][col1]
        _d2 = velRecon[_dinx][col2]
        _data = 100 * (_d2 - _d1) / np.abs(_d1)
        _data = _data.dropna()
        _dataprctnls.append([np.percentile(_data, q=q)
                             for q in [25, 50, 75]])
    return np.array(_dataprctnls)


def generate_full_summary(velReconData, nvia, params):
    # Reconstruction error as a function of duration for different via points. 
    _x = np.arange(0, len(params.Dur))
    # Create new figure.
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(3, 3, 1)
    _data = _org_by_dur(velReconData, col='sgr')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["denim blue"], alpha=0.1)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    ax.set_ylim(0, 300)
    plt.xticks(_x, params.Dur)
    ax.set_title("SGR")

    ax = fig.add_subplot(3, 3, 2)
    _data = _org_by_dur(velReconData, col='err')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["denim blue"], alpha=0.1)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    plt.xticks(_x, params.Dur)
    ax.set_ylim(0, 40)
    ax.set_title("Relative error")

    ax = fig.add_subplot(3, 3, 3)
    _data = _org_by_dur(velReconData, col='corr')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["denim blue"], alpha=0.1)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    plt.xticks(_x, params.Dur)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Correlation")

    ax = fig.add_subplot(3, 3, 4)
    _data = _org_by_dur(velReconData, col='sparc')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["denim blue"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    _data = _org_by_dur(velReconData, col='sparcs')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["pale red"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    _data = _org_by_dur(velReconData, col='sparcs-wom')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["medium green"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    ax.set_xlabel("Duration (sec)")
    plt.xticks(_x, params.Dur)
    ax.set_ylim(-3.3, -1)
    ax.set_title("SPARC")

    ax = fig.add_subplot(3, 3, 7)
    _data = _org_by_dur_reldata(velReconData, col1='sparc', col2='sparcs')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["pale red"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, color=sb.xkcd_rgb["pale red"])

    _data = _org_by_dur_reldata(velReconData, col1='sparc', col2='sparcs-wom')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["medium green"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, color=sb.xkcd_rgb["medium green"])

    ax.set_xlabel("Duration (sec)")
    plt.xticks(_x, params.Dur)
    ax.set_ylim(-100, 100)
    ax.set_title("Relative change SPARC (%)")

    ax = fig.add_subplot(3, 3, 5)
    _data = _org_by_dur(velReconData, col='ldljv')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["denim blue"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    _data = _org_by_dur(velReconData, col='ldljsv')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["pale red"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    _data = _org_by_dur(velReconData, col='ldljsv-wom')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["medium green"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5)
    ax.set_xlabel("Duration (sec)")
    plt.xticks(_x, params.Dur)
    ax.set_title("LDLJ-V")
    ax.set_ylim(-16, 4)

    ax = fig.add_subplot(3, 3, 8)
    _data = _org_by_dur_reldata(velReconData, col1='ldljv', col2='ldljsv')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["pale red"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, color=sb.xkcd_rgb["pale red"])

    _data = _org_by_dur_reldata(velReconData, col1='ldljv', col2='ldljsv-wom')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["medium green"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, color=sb.xkcd_rgb["medium green"])

    ax.set_xlabel("Duration (sec)")
    plt.xticks(_x, params.Dur)
    ax.set_ylim(-100, 100)
    ax.set_title("Relative change LDLJ-V (%)")

    ax = fig.add_subplot(3, 3, 6)
    _data = _org_by_dur(velReconData, col='ldlja')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["denim blue"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, label="Actual")
    _data = _org_by_dur(velReconData, col='ldljsa')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["pale red"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, label="Sensor")
    _data = _org_by_dur(velReconData, col='ldljsa-wom')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["medium green"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, label="W/O Mean")
    ax.set_xlabel("Duration (sec)")
    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
              fancybox=True, shadow=True, ncol=1)
    plt.xticks(_x, params.Dur)
    ax.set_title("LDLJ-A")
    ax.set_ylim(-17, -1)

    ax = fig.add_subplot(3, 3, 9)
    _data = _org_by_dur_reldata(velReconData, col1='ldlja', col2='ldljsa')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["pale red"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, color=sb.xkcd_rgb["pale red"])

    _data = _org_by_dur_reldata(velReconData, col1='ldlja', col2='ldljsa-wom')
    ax.fill_between(_x, _data[:, 0], _data[:, 2],
                    facecolor=sb.xkcd_rgb["medium green"], alpha=0.2)
    ax.plot(_x, _data[:, 1], lw=2, alpha=0.5, color=sb.xkcd_rgb["medium green"])
    ax.set_xlabel("Duration (sec)")
    plt.xticks(_x, params.Dur)
    ax.set_ylim(-100, 100)
    ax.set_title("Relative change LDLJ-A (%)")

    fig.suptitle("Summary: No. of via points: {0}".format(nvia), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


# def target_time_combs(tgts, T):
#     for i, _T in enumerate(T):
#         for j, _tgt in enumerate(tgts):
#             yield i, j, _T, _tgt


# def generate_save_simple_movements():
#     """Generates movement data and saves files to the disk.
#     """
#     allcombs = target_time_combs(Tgts, Dur)
#     for i, j, _t, _tgt in allcombs:
#         # Get movement trajectory
#         t, pos = vimu.get_mjt_pos(O[0], _tgt, _t, dt)
#         t, accl = vimu.get_mjt_accl(O[0], _tgt, _t, dt)

#         # Joint angles, end-point orientation and gravity component
#         tas = vimu.get_joint_angles(pos, L1, L2)

#         # organize all data
#         data = vimu.organize_data(t, pos, accl, grav, arm_dh, tas, dt)

#         # Save data to files.
#         _fname = "{0}/data_{1}_{2}.csv".format(outdir, i, j)
#         _str = "\rWriting Dur: {0:0.2f}s Tgt:{1:3d} | {2}"
#         sys.stdout.write(_str.format(Dur[i], j, _fname))
#         np.savetxt(_fname, data, delimiter=", ", fmt="%10.10f", header=header)

#     sys.stdout.write("\rDone                 ")


# def get_header_from_file(f):
#     _fh = open(f)
#     header = _fh.readline()
#     _header = header.split(', ')
#     _header[0] = _header[0][2:]
#     return _header


# def generate_smoothness_sgr_summary(smoothvals, params):
#     fig = plt.figure(figsize=(15, 7))
#     ax = fig.add_subplot(241)
#     plt.boxplot(smoothvals['sgr'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 )
#     plt.xticks(np.arange(1, len(params['dur']) + 1), params['dur'])
#     plt.xlabel("Movement duration (sec)")
#     plt.ylabel("Signal-to-Gravity Ratio (dB)")
#     plt.title("SGR vs. Movement duration")

#     ax = fig.add_subplot(242)
#     plt. boxplot(smoothvals['sparc'].T,
#                  notch=False,  # notch shape
#                  vert=True,  # vertical box alignment
#                  patch_artist=True,  # fill with color
#                  whis=2.5
#                  )
#     plt.boxplot(smoothvals['sparcs'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 )
#     plt.xticks(np.arange(1, len(params['dur']) + 1), params['dur'])
#     plt.xlabel("Movement duration (sec)")
#     plt.ylabel("SPARC Smoothness")
#     plt.title("SPARC vs. Movement duration")

#     ax = fig.add_subplot(246)
#     plt. boxplot(smoothvals['sparc'].T,
#                  notch=False,  # notch shape
#                  vert=True,  # vertical box alignment
#                  patch_artist=True,  # fill with color
#                  whis=2.5
#                  )
#     plt.boxplot(smoothvals['sparcs-wom'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 )
#     plt.xticks(np.arange(1, len(params['dur']) + 1), params['dur'])
#     plt.xlabel("Movement duration (sec)")
#     plt.ylabel("SPARC Smoothness")
#     plt.title("SPARC (WOM) vs. Movement duration")

#     ax = fig.add_subplot(243)
#     plt.boxplot(smoothvals['ldljv'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 whis=2.5
#                 )
#     plt.boxplot(smoothvals['ldljsv'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 )
#     plt.xticks(np.arange(1, len(params['dur']) + 1), params['dur'])
#     ax.set_ylim(-6, 6)
#     plt.xlabel("Movement duration (sec)")
#     plt.ylabel("LDLJ-V Smoothness")
#     plt.title("LDLJ-V vs. Movement duration")

#     ax = fig.add_subplot(247)
#     plt.boxplot(smoothvals['ldljv'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 whis=2.5
#                 )
#     plt.boxplot(smoothvals['ldljsv-wom'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 )
#     plt.xticks(np.arange(1, len(params['dur']) + 1), params['dur'])
#     ax.set_ylim(-6, 6)
#     plt.xlabel("Movement duration (sec)")
#     plt.ylabel("LDLJ-V Smoothness")
#     plt.title("LDLJ-V (WOM) vs. Movement duration")

#     ax = fig.add_subplot(244)
#     plt.boxplot(smoothvals['ldlja'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 whis=2.5
#                 )
#     plt.boxplot(smoothvals['ldljsa'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 )
#     plt.xticks(np.arange(1, len(params['dur']) + 1), params['dur'])
#     ax.set_ylim(-6, 6)
#     plt.xlabel("Movement duration (sec)")
#     plt.ylabel("LDLJ-A Smoothness")
#     plt.title("LDLJ-A vs. Movement duration")

#     ax = fig.add_subplot(248)
#     plt.boxplot(smoothvals['ldlja'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 whis=2.5
#                 )
#     plt.boxplot(smoothvals['ldljsa-wom'].T,
#                 notch=False,  # notch shape
#                 vert=True,  # vertical box alignment
#                 patch_artist=True,  # fill with color
#                 )
#     plt.xticks(np.arange(1, len(params['dur']) + 1), params['dur'])
#     ax.set_ylim(-6, 6)
#     plt.xlabel("Movement duration (sec)")
#     plt.ylabel("LDLJ-A Smoothness")
#     plt.title("LDLJ-A (WOM) vs. Movement duration")

#     plt.tight_layout()

#     return fig
