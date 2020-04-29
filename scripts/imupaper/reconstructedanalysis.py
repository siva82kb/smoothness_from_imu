"""Module containing the necessary variables and functions for analysing
smoothness from reconstructed IMU data for complex reaching movements.

The reconstruction involves estimating IMU orientation, which is then used
to correct for the orientation and the effect of gravity.

Author: Sivakumar Balasubramanian
Date: Sometime in 2019 :)
"""

import numpy as np
import os
import json
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import pandas as pd
import seaborn as sb
import progressbar as pb
import itertools
from scipy import signal
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.append("scripts")
import myrobotics as myrob
from imupaper import virtualimu as vimu
from smoothness import sparc
from smoothness import log_dimensionless_jerk
from smoothness import log_dimensionless_jerk_imu
from smoothness import log_dimensionless_jerk_factors
from smoothness import log_dimensionless_jerk_imu_factors


def GetParamDetailsForFile(f):
    _nv, _nr = f.split('.csv')[0].split('/')[-1].split('via_')
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
    dt = 0.001
    fs = 1 / dt
    # Movement amplitude
    Amp = 15.0
    # Different movement duration.
    Dur = [2.5, 5.0, 10.0, 20.0]
    Ndur = len(Dur)

    # Different levels of reconstruction errors.
    phiErr = [5.0, 25.0, 50.0]
    NphiErrRep = 5

    # Data related stuff.
    datadir = "data/mjtdata"
    files = glob.glob("{0}/*.csv".format(datadir))
    # Get parameters
    nVia = [1, 2, 5, 10]
    NViaRep = 25

    # Total number of reps.
    Ntotal = len(nVia) * len(Dur) * len(phiErr) * NViaRep * NphiErrRep

    # gravity
    grav = np.array([[0, 0, -98.1]]).T

    # High pass filter cut offs
    fc = [0.05, 0.1, 0.25, 0.5]
    ford = [1, 2, 3, 4]
    fdet = [["pad", "odd"], ["pad", "even"], ["pad", "constant"],
            ["gust", None]]
    fcombs = [_c for _c in itertools.product(fc, ford, fdet)] + ["wom"]

    # Perturbation angle filter parameter.
    # Nf = 1 + int(0.5 / dt)
    Tf = 0.5

    # out directory
    # outdir = "../data/reconstructed/"
    outdir = "data/reconstructed/"
    fnamestr = "_".join(("{0}/rdata_v{{0:02d}}", "d{{1:02d}}",
                         "pe{{2:02d}}", "vr{{3:02d}}", "pr{{4:02d}}.csv"))
    fnamestr = fnamestr.format(outdir)
    # Data header.
    headcols = ("time",
                "ax", "ay", "az",
                "vx", "vy", "vz",
                "phi1", "phi2", "phi3",
                "ep1", "ep2", "ep3", "ep4",
                "rnorm",
                "axs", "ays", "azs",
                "gxs", "gys", "gzs",
                "axr", "ayr", "azr",
                "vxr", "vyr", "vzr",
                "axr-wom", "ayr-wom", "azr-wom",
                "vxr-wom", "vyr-wom", "vzr-wom",
                "spd", "spdr", "spdr-wom",
                "gyx", "gyy", "gyz",
                "gyxr", "gyyr", "gyzr")
    # combs = itertools.product(('accl', 'vel'), range(4), ('x', 'y', 'z'))
    # headcols += tuple(["{0}hp{1}{2}".format(i, j, k)  for i, j, k in combs])
    # headcols += tuple(["spdrhp{0}".format(i) for i in range(4)])
    header = ", ".join(headcols)

    @staticmethod
    def write_params_file():
        # Details of the simulated movements
        params = {'Amp': Params.Amp,
                  'gravity': Params.grav.tolist(),
                  'dur': Params.Dur,
                  'phierr': Params.phiErr,
                  'dt': Params.dt,
                  'datadir': Params.datadir,
                  'files': Params.files,
                  'nVia': Params.nVia,
                  'Nrep': Params.NViaRep,
                  'NphiErrRep': Params.NphiErrRep,
                  'Ntotal': Params.Ntotal,
                  'fc': Params.fc, 'ford': Params.ford,
                  'fdetails': Params.fdet,
                  'fcombs': Params.fcombs}
        # Write the parameter file
        # Write params file
        with open("{0}/params.json".format(Params.outdir), "w") as fh:
            json.dump(params, fh, indent=4)


def generate_perturbation_angles(N, Nf, philims, offset=True):
    """Three Euler angles are generated as a cummulative sum of random Gaussian noise,
    which is then filtering by a moving average filter of width 0.5s."""
    phi1 = signal.savgol_filter(np.cumsum(np.random.randn(N)), Nf, 0, mode='mirror')
    phi1 = (philims[0] * phi1 / np.max(np.abs(phi1)))
    if offset:
        phi1 += 0.1 * np.random.randn(1)
    phi2 = signal.savgol_filter(np.cumsum(np.random.randn(N)), Nf, 0, mode='mirror')
    phi2 = (philims[1] * phi2 / np.max(np.abs(phi2)))
    if offset:
        phi2 += 0.1 * np.random.randn(1)
    phi3 = signal.savgol_filter(np.cumsum(np.random.randn(N)), Nf, 0, mode='mirror')
    phi3 = (philims[2] * phi3 / np.max(np.abs(phi3)))
    if offset:
        phi3 += 0.1 * np.random.randn(1)
    return np.array([phi1, phi2, phi3])


def get_perturbation_rotmat(p):
    return np.matmul(np.matmul(myrob.rotz(p[0]), myrob.roty(p[1])),
                     myrob.rotx(p[2]))


def get_all_indices(params):
    return itertools.product(range(len(params.nVia)),
                             range(len(params.Dur)),
                             range(len(params.phiErr)),
                             range(params.NViaRep),
                             range(params.NphiErrRep))


def get_all_movement_data(params):
    # File name format.
    fname_str = "{0}/{{0:02d}}via_{{1:04d}}_new.csv".format(params.datadir)
    inxs = get_all_indices(params)
    _fac = np.pi / 180.
    for inx in inxs:
        nv = params.nVia[inx[0]]
        dur = params.Dur[inx[1]]
        perr = params.phiErr[inx[2]]

        f = fname_str.format(nv, inx[3])
        # Read file.
        if os.path.isfile(f):
            data = pd.read_csv(f)
            yield {"inx": inx, "nv": nv, "dur": dur,
                   "perr": _fac * perr * np.ones(3),
                   "N": len(data), "data": data, "fname": f}
        else:
            continue


def generate_save_reconstructed_data(params, offset=True):
    # define progress bar
    widgets = ['[', pb.Timer(), '] ', pb.Bar(), ' (', pb.ETA(), ')']
    bar = pb.ProgressBar(widgets=widgets, maxval=params.Ntotal)

    # Get generator for all data.
    _accinx = ['ax', 'ay', 'az']
    _velinx = ['vx', 'vy', 'vz']
    _t = np.arange(0, 1.0, params.dt)
    alldata = get_all_movement_data(params)
    cnt = 0
    for d in alldata:
        cnt += 1
        # Get acceleration and linear velocity data.
        _k = params.Amp / np.power(d['dur'], 2.0)
        accl = _k * np.array(d['data'][_accinx])
        _k = params.Amp / d['dur']
        vel = _k * np.array(d['data'][_velinx])
        spd = np.linalg.norm(vel, axis=1).reshape((d['N'], 1))

        # 1. Generate the reconstruction error data.
        _Nf = int(params.Tf / (params.dt * d['dur']))
        _Nf += 1 if (_Nf % 2) == 0 else 0
        phiErr = generate_perturbation_angles(d['N'], _Nf, d['perr'], offset)
        # 2a. Get the Euler parameters corresponding to the reconstruction
        # errors.
        ep = [myrob.rotmat_to_eulerparam(rotm=get_perturbation_rotmat(p))
              for p in phiErr.T]
        # 2b. Euler parameter to angle axis.
        _anax = np.array([myrob.euleraram_to_angaxis(_ep) for _ep in ep])
        _dphi = np.hstack(([0], np.diff(_anax[:, 0])))
        _dphi = (1 / (d['dur'] * params.dt)) * _dphi
        # 3. We simply se the velocity as the gyro signal. This is only for
        # analysis purpose.
        gyro = vel

        # Calculate IMU data.
        accls = [0] * d['N']
        gravs = [0] * d['N']
        acclr = [0] * d['N']
        velr = [0] * d['N']
        gyror = [0] * d['N']
        _rnorm = [0] * d['N']
        for i in range(d['N']):
            # Residual rotation matrix.
            _R = get_perturbation_rotmat(p=phiErr[:, i])
            # Reconstructed acceleration
            _ag = accl.T[:, i].reshape((3, 1)) + params.grav
            accls[i] = np.matmul(_R, _ag)[:, 0]
            gravs[i] = np.matmul(_R, params.grav)[:, 0]
            acclr[i] = accls[i] - params.grav[:, 0]
            # Reconstructed gyroscope data
            gyror[i] = np.matmul(_R, gyro[i].reshape(3, 1)).T[0]
            # Reconstruction error 2-norm
            _rnorm[i] = [np.linalg.norm(_R - np.eye(3), ord=2)]

        # Format into numpy array
        t = _t.reshape((d['N'], 1)) * d['dur']
        ep = np.array(ep)
        accls = np.array(accls)
        gravs = np.array(gravs)
        acclr = np.array(acclr)
        velr = np.cumsum(acclr, axis=0) * params.dt * d['dur']
        spdr = np.linalg.norm(velr, axis=1).reshape((d['N'], 1))
        gyror = np.array(gyror)
        _rnorm = np.array(_rnorm)

        # Calculate variables without mean
        acclr_wom = acclr - np.mean(acclr, axis=0)
        velr_wom = np.cumsum(acclr_wom, axis=0) * params.dt * d['dur']
        spdr_wom = np.linalg.norm(velr_wom, axis=1).reshape((d['N'], 1))

        # Write data to a csv file.
        _fname = params.fnamestr.format(*d['inx'])
        _data = np.hstack((t, accl, vel, phiErr.T, ep, _rnorm,
                           accls, gravs, acclr, velr,
                           acclr_wom, velr_wom, spd, spdr,
                           spdr_wom, gyro, gyror))
        np.savetxt(_fname, _data, delimiter=",", fmt="%10.10f",
                   header=params.header)
        bar.update(cnt)


def _smoothsparc(sp, spr, sprwom, fs):
    _ss, _, _ = sparc(sp, fs)
    _ssr, _, _ = sparc(spr, fs)
    _ssrwom, _, _ = sparc(sprwom, fs)
    return _ss, _ssr, _ssrwom


def _smoothsparcgyro(gyro, gyror, fs):
    _ss, _, _ = sparc(gyro, fs)
    _ssr, _, _ = sparc(gyror, fs)
    return _ss, _ssr


def _smoothldljvgyro(gyro, gyror, fs):
    _sl = log_dimensionless_jerk(gyro, fs=fs, data_type="vel")
    _slr = log_dimensionless_jerk(gyror, fs=fs, data_type="vel")
    return _sl, _slr


def _smoothldljvgyro_factors(gyro, gyror, fs):
    _sl = log_dimensionless_jerk_factors(gyro, fs=fs, data_type="vel")
    _slr = log_dimensionless_jerk_factors(gyror, fs=fs, data_type="vel")
    return _sl, _slr


def _smoothldljv(v, vr, vrwom, fs):
    _sl = log_dimensionless_jerk(v, fs=fs, data_type="vel")
    _slr = log_dimensionless_jerk(vr, fs=fs, data_type="vel")
    _slrwom = log_dimensionless_jerk(vrwom, fs=fs, data_type="vel")
    return _sl, _slr, _slrwom


def _smoothldlja(ac, acr, fs):
    _sl = log_dimensionless_jerk(ac, fs=fs, data_type="accl", rem_mean=True)
    _slr = log_dimensionless_jerk(acr, fs=fs, data_type="accl", rem_mean=False)
    _slrwom = log_dimensionless_jerk(acr, fs=fs, data_type="accl", rem_mean=True)
    return _sl, _slr, _slrwom


def _smoothldljimu(ac, acr, gyror, grav, fs):
    _sl = log_dimensionless_jerk_imu(ac, None, grav, fs)
    _slr = log_dimensionless_jerk_imu(acr + grav.T, gyror, grav, fs)
    return _sl, _slr


def _smoothldljv_factors(v, vr, vrwom, fs):
    _sfl = log_dimensionless_jerk_factors(v, fs=fs, data_type="vel")
    _sflr = log_dimensionless_jerk_factors(vr, fs=fs, data_type="vel")
    _sflrwom = log_dimensionless_jerk_factors(vrwom, fs=fs, data_type="vel")
    return _sfl, _sflr, _sflrwom


def _smoothldlja_factors(ac, acr, fs):
    _sfl = log_dimensionless_jerk_factors(ac, fs=fs, data_type="accl", rem_mean=True)
    _sflr = log_dimensionless_jerk_factors(acr, fs=fs, data_type="accl", rem_mean=False)
    _sflrwom = log_dimensionless_jerk_factors(acr, fs=fs, data_type="accl", rem_mean=True)
    return _sfl, _sflr, _sflrwom

def _smoothldljaimu(acs, grav, fs):
    _sfl1 = ldlja_imu(acs, grav, fs, scale="ap1")
    _sfl2 = ldlja_imu(acs, grav, fs, scale="ap2")
    _sfl3 = ldlja_imu(acs, grav, fs, scale="ap2")
    return _sfl1, _sfl2, _sfl3


def _smoothldljaimu_factors(acs, grav, fs):
    _sfl1 = ldlja_imu_factors(acs, grav, fs, scale="ap1")
    _sfl2 = ldlja_imu_factors(acs, grav, fs, scale="ap2")
    _sfl3 = ldlja_imu_factors(acs, grav, fs, scale="ap3")
    return _sfl1, _sfl2, _sfl3


def _smoothldljimu_factors(ac, acr, gyror, grav, fs):
    _sfl = log_dimensionless_jerk_imu_factors(ac, None, grav, fs)
    _sflr = log_dimensionless_jerk_imu_factors(acr + grav.T, gyror, grav, fs)
    return _sfl, _sflr


def generate_reconst_traj(params, save_indiv_plots=False):
    N = len(params.nVia) * len(params.phiErr)
    cnt = 0
    # define progress bar
    widgets = ['[', pb.Timer(), '] ', pb.Bar(), ' (', pb.ETA(), ')']
    bar = pb.ProgressBar(widgets=widgets, maxval=N)
    _fnamestr = "{0}summary/traj/".format(params.outdir)
    # Go through each via point and generate a individual PDF file.
    for via, nv in enumerate(params.nVia):
        _fname = "{0}/traj_v{1:02d}".format(_fnamestr, nv)
        with PdfPages("{0}.pdf".format(_fname)) as pdf:
            # A page for each level of reconstructon error
            for perr, _ in enumerate(params.phiErr):
                fig = gen_reconst_traj_for_viaperr(via, perr, params)
                pdf.savefig(fig)
                # Individual SVG and PNG files
                if save_indiv_plots:
                    _fnameperr = "{0}_p{1:02d}.{2}"
                    # fig.savefig(_fnameperr.format(_fname, perr, "pdf"), format="pdf")
                    fig.savefig(_fnameperr.format(_fname, perr, "svg"), format="svg")
                plt.close()
                cnt += 1
                bar.update(cnt)

def gen_reconst_traj_for_viaperr(via, perr, params):
    # define progress bar
    widgets = ['[', pb.Timer(), '] ', pb.Bar(), ' (', pb.ETA(), ')']
    bar = pb.ProgressBar(widgets=widgets, maxval=params.Ntotal)

    indices = get_all_indices(params)
    cnt = 0

    fig = plt.figure(figsize=(20, 12))
    row, col = 5, len(params.Dur)
    axs = [fig.add_subplot(row, col, i+1)
           for i in range(row * col)]
    for inx in indices:
        cnt += 1
        # Plot only the correction via point and
        # reconstruction error
        if inx[0] != via or inx[2] != perr:
            continue

        # Does the file exist?
        _f = params.fnamestr.format(*inx)
        if os.path.isfile(_f) is False:
            continue

        # The file exists.
        # read reconstrucuted data
        _f = params.fnamestr.format(*inx)
        data = pd.read_csv(filepath_or_buffer=_f, delimiter=',')
        data = data.rename(columns=lambda x: x.strip())
        data = data.rename(columns={"# time": "time"})
        _N = len(data)

        # Reconstruction error angles..
        _axinx = inx[1] + 1
        axs[_axinx - 1].plot(data['time'], (180. / np.pi) * data['phi1'],
                             color='b', alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], (180. / np.pi) * data['phi2'],
                             color='r', alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], (180. / np.pi) * data['phi3'],
                             color='k', alpha=0.2, lw=1)
        axs[_axinx - 1].set_ylim(-100, 100)

        # Acceleration.
        _axinx = col + inx[1] + 1
        axs[_axinx - 1].plot(data['time'], data['axr'], color='b',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['ayr'], color='b',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['azr'], color='b',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['axr-wom'], color='r',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['ayr-wom'], color='r',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['ayr-wom'], color='r',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['ax'], color='k',
                             alpha=0.7, lw=1)
        axs[_axinx - 1].plot(data['time'], data['ay'], color='k',
                             alpha=0.7, lw=1)
        axs[_axinx - 1].plot(data['time'], data['az'], color='k',
                             alpha=0.7, lw=1)

        # Velocty reconstructed with and without mean
        _axinx = 2 * col + inx[1] + 1
        axs[_axinx - 1].plot(data['time'], data['vxr'], color='b',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vyr'], color='b',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vzr'], color='b',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vxr-wom'], color='r',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vyr-wom'], color='r',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vyr-wom'], color='r',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vx'], color='k',
                             alpha=0.7, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vy'], color='k',
                             alpha=0.7, lw=1)
        axs[_axinx - 1].plot(data['time'], data['vz'], color='k',
                             alpha=0.7, lw=1)

        # Speed reconstructed with mean
        _axinx = 3 * col + inx[1] + 1
        axs[_axinx - 1].plot(data['time'], data['spdr'], color='b',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['spd'], color='k',
                             alpha=0.7, lw=1)

        # Speed reconstructed without mean
        _axinx = 4 * col + inx[1] + 1
        axs[_axinx - 1].plot(data['time'], data['spdr-wom'], color='r',
                             alpha=0.2, lw=1)
        axs[_axinx - 1].plot(data['time'], data['spd'], color='k',
                             alpha=0.7, lw=1)
        bar.update(cnt)
    bar.update(cnt)
    # Set labels.
    for i in range(10, row * col):
        axs[i].set_xlabel("Time (sec)")
    for i, l in enumerate(["Angle error", "Accl.", "Vel.", "Speed"]):
        axs[col * i].set_ylabel(l)
    _titlestr = "Trajectories for different duration (Nv: {0}; Nperr: {1})"
    fig.suptitle(_titlestr.format(via, params.phiErr[perr]), fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def get_smoothness_perf(params):
    _sz = (len(params.nVia), len(params.Dur), len(params.phiErr),
           params.NViaRep, params.NphiErrRep)
    cols = ["via", "dur", "phierr", "viarep", "perrep",
            "sgr", "agr", "err", "corr", "velnorm",
            "angvelmax", "angrange", "angvelrms", "rnorm",
            "sparc", "sparcr", "sparcr-wom",
            "ldljv", "ldljrv", "ldljrv-wom",
            "ldlja", "ldljra", "ldljra-wom",
            "ldlja1", "ldlja2", "ldlja3",
            "ldljimu", "ldljrimu",
            "ldljv-T", "ldljv-A", "ldljv-J",
            "ldljrv-T", "ldljrv-A", "ldljrv-J",
            "ldljrv-wom-T", "ldljrv-wom-A", "ldljrv-wom-J",
            "ldlja-T", "ldlja-A", "ldlja-J",
            "ldljra-T", "ldljra-A", "ldljra-J",
            "ldljra-wom-T", "ldljra-wom-A", "ldljra-wom-J",
            "ldlja1-T", "ldlja1-A", "ldlja1-J",
            "ldlja2-T", "ldlja2-A", "ldlja2-J",
            "ldlja3-T", "ldlja3-A", "ldlja3-J",
            "sparc-gyro", "sparcr-gyro",
            "ldljv-gyro", "ldljvr-gyro",
            "ldljv-gyro-T", "ldljv-gyro-A", "ldljv-gyro-J",
            "ldljvr-gyro-T", "ldljvr-gyro-A", "ldljvr-gyro-J",]
    velRecon = pd.DataFrame(columns=cols)

    # define progress bar
    widgets = ['[', pb.Timer(), '] ', pb.Bar(), ' (', pb.ETA(), ')']
    bar = pb.ProgressBar(widgets=widgets, maxval=params.Ntotal)

    _phinx = ['phi1', 'phi2', 'phi3']
    _accinx = ['ax', 'ay', 'az']
    _accsinx = ['axs', 'ays', 'azs']
    _accrinx = ['axr', 'ayr', 'azr']
    _accrwominx = ['axr-wom', 'ayr-wom', 'azr-wom']
    _velinx = ['vx', 'vy', 'vz']
    _velrinx = ['vxr', 'vyr', 'vzr']
    _velrwominx = ['vxr-wom', 'vyr-wom', 'vzr-wom']
    _gyinx = ['gyx', 'gyy', 'gyz']
    _gyrinx = ['gyxr', 'gyyr', 'gyzr']

    indices = get_all_indices(params)
    cnt = 0
    for inx in indices:
        _f = Params.fnamestr.format(*inx)
        if os.path.isfile(_f) is True:
            # read reconstrucuted data
            data = pd.read_csv(filepath_or_buffer=_f, delimiter=',')
            data = data.rename(columns=lambda x: x.strip())
            _N = len(data)

            # Max. angular velocity.
            _avmax = np.max(np.linalg.norm(np.array(data[_gyinx]), axis=1))

            # Angular error range.
            _angrange = np.max(np.max(np.array(data[_phinx]), axis=0) -
                               np.min(np.array(data[_phinx]), axis=0))

            # RMS of angular velocity
            _N = np.sqrt(np.shape(np.array(data[_gyinx]))[0])
            _avrms = np.linalg.norm(np.array(data[_gyinx])) / _N

            # Reconstruction error, 2-norm.
            _rnorm = np.linalg.norm(data['rnorm']) / _N

            # Sensor to gravity ratio
            _g = _N * np.linalg.norm(params.grav)
            _s = np.linalg.norm(np.array(data[_accrinx]) + params.grav.T)
            _a = np.linalg.norm(np.array(data[_accrinx]))
            _sgr = _s / _g
            _agr = _a / _g

            # Speed reconstruction error
            _temp = np.max(np.abs(data['spd'] - data['spdr-wom']))
            _temp1 = np.max(data['spd'])
            _err = _temp / _temp1

            # Speed correlation
            _corr = np.corrcoef(data['spd'], data['spdr-wom'])[0, 1]

            # Smoothness analysis
            _fs = Params.fs / params.Dur[inx[1]]
            # SPARC
            _sparc = _smoothsparc(np.array(data['spd']),
                                  np.array(data['spdr']),
                                  np.array(data['spdr-wom']), _fs)
            # LDLJ velocity
            _ldljv = _smoothldljv(np.array(data[_velinx]),
                                  np.array(data[_velrinx]),
                                  np.array(data[_velrwominx]), _fs)
            # LDLJV terms
            _ldljvfac = _smoothldljv_factors(np.array(data[_velinx]),
                                             np.array(data[_velrinx]),
                                             np.array(data[_velrwominx]), _fs)
            # LDLJ acceleration
            _ldlja = _smoothldlja(np.array(data[_accinx]),
                                  np.array(data[_accrinx]), _fs)
            # LDLJA terms
            _ldljafac = _smoothldlja_factors(np.array(data[_accinx]),
                                             np.array(data[_accrinx]), _fs)
            
            # LDLJ IMU acceleration with different apeaks
            _ldljaimu = _smoothldljaimu(np.array(data[_accsinx]), params.grav,
                                        _fs)

            # LDLJA terms
            _ldljaimufac = _smoothldljaimu_factors(np.array(data[_accsinx]),
                                                   params.grav, _fs)
            
            # LDLJ IMU
            _ldljimu =    _smoothldljimu(np.array(data[_accinx]),
                                         np.array(data[_accrinx]),
                                         np.array(data[_gyinx]),
                                         params.grav, _fs)
            # LDLJA terms
            _ldljimufac = _smoothldljimu_factors(np.array(data[_accinx]),
                                                 np.array(data[_accrinx]),
                                                 np.array(data[_gyinx]),
                                                 params.grav, _fs)
            
            # SPARC on gryo data
            _sparcgyro = _smoothsparcgyro(np.linalg.norm(data[_gyinx], axis=1),
                                          np.linalg.norm(data[_gyrinx], axis=1), _fs)
            
            # LDLJV on gryo data 
            _ldljvgyro = _smoothldljvgyro(np.array(data[_gyinx]),
                                          np.array(data[_gyrinx]), _fs)
            
            # LDLJV terms on gryo data 
            _ldljvgyrofac = _smoothldljvgyro_factors(np.array(data[_gyinx]),
                                                     np.array(data[_gyrinx]), _fs)

            # Append to dataframe
            _data = {"via": [params.nVia[inx[0]]], "dur": params.Dur[inx[1]],
                     "phierr": [params.phiErr[inx[2]]],
                     "viarep": inx[3], "perrep": inx[4],
                     "sgr": [_sgr], "agr": [_agr], 
                     "err": [_err], "corr": [_err],
                     "angvelmax": [_avmax], "angrange": [_angrange],
                     "angvelrms": [_avrms],
                     "rnorm": [_rnorm],
                     "velnorm": [np.linalg.norm(data[_gyinx]) / np.sqrt(_N)],
                     "sparc": [_sparc[0]], "sparcr": [_sparc[1]],
                     "sparcr-wom": [_sparc[2]],
                     "ldljv": [_ldljv[0]],
                     "ldljrv": [_ldljv[1]],
                     "ldljrv-wom": [_ldljv[2]],
                     "ldljv-T": [_ldljvfac[0][0]],
                     "ldljv-A": [_ldljvfac[0][1]],
                     "ldljv-J": [_ldljvfac[0][2]],
                     "ldljrv-T": [_ldljvfac[1][0]],
                     "ldljrv-A": [_ldljvfac[1][1]],
                     "ldljrv-J": [_ldljvfac[1][2]],
                     "ldljrv-wom-T": [_ldljvfac[2][0]],
                     "ldljrv-wom-A": [_ldljvfac[2][1]],
                     "ldljrv-wom-J": [_ldljvfac[2][2]],
                     "ldlja": [_ldlja[0]],
                     "ldljra": [_ldlja[1]],
                     "ldljra-wom": [_ldlja[2]],
                     "ldljimu": [_ldljimu[0]],
                     "ldljrimu": [_ldljimu[1]],
                     "ldlja-T": [_ldljafac[0][0]],
                     "ldlja-A": [_ldljafac[0][1]],
                     "ldlja-J": [_ldljafac[0][2]],
                     "ldljra-T": [_ldljafac[1][0]],
                     "ldljra-A": [_ldljafac[1][1]],
                     "ldljra-J": [_ldljafac[1][2]],
                     "ldljra-wom-T": [_ldljafac[2][0]],
                     "ldljra-wom-A": [_ldljafac[2][1]],
                     "ldljra-wom-J": [_ldljafac[2][2]],
                     "ldlja1": [_ldljaimu[0]],
                     "ldlja1-T": [_ldljaimufac[0][0]],
                     "ldlja1-A": [_ldljaimufac[0][1]],
                     "ldlja1-J": [_ldljaimufac[0][2]],
                     "ldlja2": [_ldljaimu[1]],
                     "ldlja2-T": [_ldljaimufac[1][0]],
                     "ldlja2-A": [_ldljaimufac[1][1]],
                     "ldlja2-J": [_ldljaimufac[1][2]],
                     "ldlja3": [_ldljaimu[2]],
                     "ldlja3-T": [_ldljaimufac[2][0]],
                     "ldlja3-A": [_ldljaimufac[2][1]],
                     "ldlja3-J": [_ldljaimufac[2][2]],
                     "sparc-gyro": [_sparcgyro[0]],
                     "sparcr-gyro": [_sparcgyro[1]],
                     "ldljv-gyro": [_ldljvgyro[0]],
                     "ldljv-gyro-T": [_ldljvgyrofac[0][0]],
                     "ldljv-gyro-A": [_ldljvgyrofac[0][1]],
                     "ldljv-gyro-J": [_ldljvgyrofac[0][2]],
                     "ldljvr-gyro": [_ldljvgyro[1]],
                     "ldljvr-gyro-T": [_ldljvgyrofac[1][0]],
                     "ldljvr-gyro-A": [_ldljvgyrofac[1][1]],
                     "ldljvr-gyro-J": [_ldljvgyrofac[1][2]],
                     }
            velRecon = velRecon.append(pd.DataFrame.from_dict(_data),
                                       ignore_index=True)
        cnt += 1
        bar.update(cnt)
    return velRecon


def get_chopping_index(accsr, choprcnt):
    _accr_norm = np.linalg.norm(accsr, axis=1)
    _accr_chopinx = 1.0 * (_accr_norm >= (choprcnt / 100.) * np.max(_accr_norm))
    _diff_inx = np.diff(_accr_chopinx)
    # Find the first 0 to 1 transition.
    if _accr_chopinx[0] == 1:
      _strtinx = 0
    else:
      _strtinx = 1 + np.where(_diff_inx == 1)[0][0]

    # Find the last 1 to 0 transition.
    if _accr_chopinx[-1] == 1:
      _stpinx = len(accsr)
    else:
      _stpinx = np.where(_diff_inx == -1)[0][-1]
    
    return _strtinx, _stpinx


def get_smoothness_perf_with_chopping(params, choparam=[0, 1, 5, 10]):
    _sz = (len(params.nVia), len(params.Dur), len(params.phiErr),
           params.NViaRep, params.NphiErrRep)
    cols = ["via", "dur", "phierr", "viarep", "perrep",
            "sgr", "agr", "velnorm",
            "angvelmax", "angrange", "angvelrms", "rnorm",
            "choprcnt",
            "ldlja", "ldlja-chop",
            "ldljra", "ldljra-chop",
            "ldljra-wom", "ldljra-wom-chop"]
    velRecon = pd.DataFrame(columns=cols)

    # define progress bar
    widgets = ['[', pb.Timer(), '] ', pb.Bar(), ' (', pb.ETA(), ')']
    bar = pb.ProgressBar(widgets=widgets, maxval=params.Ntotal * len(choparam))

    _phinx = ['phi1', 'phi2', 'phi3']
    _accinx = ['ax', 'ay', 'az']
    _accrinx = ['axr', 'ayr', 'azr']
    _accrwominx = ['axr-wom', 'ayr-wom', 'azr-wom']
    _gyinx = ['gyx', 'gyy', 'gyz']

    indices = get_all_indices(params)
    cnt = 0
    for inx in indices:
        _f = params.fnamestr.format(*inx)
        #  Do nothing if the file is not found.
        if os.path.isfile(_f) is False:
            cnt += 1
            bar.update(cnt)
            continue
        # File is available.
        # read reconstrucuted data
        data = pd.read_csv(filepath_or_buffer=_f, delimiter=',')
        data = data.rename(columns=lambda x: x.strip())
        
        # Go through different levels of chopping
        for _c in choparam:
            # Chop accelerometer data
            _strtinx, _stpinx = get_chopping_index(data[_accrinx], _c)
            # sys.stdout.write(f"\r{_f} - {_c:02d} - {_strtinx:4d}:{_stpinx:04d}  ")
            # sys.stdout.flush()

            # Max. angular velocity.
            _avmax = np.max(np.linalg.norm(np.array(data[_gyinx]), axis=1))

            # Angular error range.
            _angrange = np.max(np.max(np.array(data[_phinx]), axis=0) -
                                np.min(np.array(data[_phinx]), axis=0))

            # RMS of angular velocity
            _N = np.sqrt(np.shape(np.array(data[_gyinx]))[0])
            _avrms = np.linalg.norm(np.array(data[_gyinx])) / _N

            # Reconstruction error, 2-norm.
            _rnorm = np.linalg.norm(data['rnorm']) / _N

            # Sensor to gravity ratio
            _g = _N * np.linalg.norm(params.grav)
            _s = np.linalg.norm(np.array(data[_accrinx]) + params.grav.T)
            _a = np.linalg.norm(np.array(data[_accrinx]))
            _sgr = _s / _g
            _agr = _a / _g

            # Smoothness analysis
            _fs = params.fs / params.Dur[inx[1]]
            # LDLJ acceleration
            _ldlja = _smoothldlja(np.array(data[_accinx]),
                                  np.array(data[_accrinx]),
                                     np.array(data[_accrwominx]), _fs)
            # LDLJ acceleration chop
            _ldlja_chop = _smoothldlja(np.array(data[_accinx][_strtinx:_stpinx]),
                                       np.array(data[_accrinx][_strtinx:_stpinx]),
                                          np.array(data[_accrwominx][_strtinx:_stpinx]), _fs)

            # Append to dataframe
            _data = {"via": [params.nVia[inx[0]]], "dur": params.Dur[inx[1]],
                     "phierr": [params.phiErr[inx[2]]],
                     "viarep": inx[3], "perrep": inx[4],
                     "sgr": [_sgr], "agr": [_agr], 
                     "angvelmax": [_avmax], "angrange": [_angrange],
                     "angvelrms": [_avrms],
                     "rnorm": [_rnorm],
                     "choprcnt": [_c],
                     "ldlja": [_ldlja[0]],
                     "ldljra": [_ldlja[1]],
                     "ldljra-wom": [_ldlja[2]],
                     "ldlja-chop": [_ldlja_chop[0]],
                     "ldljra-chop": [_ldlja_chop[1]],
                     "ldljra-wom-chop": [_ldlja_chop[2]],
                      }
            velRecon = velRecon.append(pd.DataFrame.from_dict(_data),
                                        ignore_index=True)
            cnt += 1
            bar.update(cnt)
    return velRecon


def plot_ldlja_chopping_summary(velRecon_chop):
    fig = plt.figure(figsize=(10, 6.7))
    ax = fig.add_subplot(211)
    _inx0 = velRecon_chop['choprcnt'] == 0.0
    _inx1 = velRecon_chop['choprcnt'] == 1.0
    _inx5 = velRecon_chop['choprcnt'] == 5.0
    _inx10 = velRecon_chop['choprcnt'] == 10.0
    _err0 = (velRecon_chop[_inx0]['ldljra-chop'] - velRecon_chop[_inx0]['ldlja']) / np.abs(velRecon_chop[_inx0]['ldlja'])
    _err1 = (velRecon_chop[_inx1]['ldljra-chop'] - velRecon_chop[_inx1]['ldlja']) / np.abs(velRecon_chop[_inx1]['ldlja'])
    _err5 = (velRecon_chop[_inx5]['ldljra-chop'] - velRecon_chop[_inx5]['ldlja']) / np.abs(velRecon_chop[_inx5]['ldlja'])
    _err10 = (velRecon_chop[_inx10]['ldljra-chop'] - velRecon_chop[_inx10]['ldlja']) / np.abs(velRecon_chop[_inx10]['ldlja'])

    hist0, bin_edges0 = np.histogram(_err0, bins=100, density=True)
    hist1, bin_edges1 = np.histogram(_err1, bins=100, density=True)
    hist5, bin_edges5 = np.histogram(_err5, bins=100, density=True)
    hist10, bin_edges10 = np.histogram(_err10, bins=100, density=True)

    ax.plot(bin_edges0[1:], hist0, color='0.7', lw=4, label="0%")
    ax.plot(bin_edges1[1:], hist1, lw=2, label="1%")
    ax.plot(bin_edges5[1:], hist5, lw=2, label="5%")
    ax.plot(bin_edges10[1:], hist10, lw=2, label="10%")
    ax.legend(loc=2, fontsize=14)
    # xlabel("Relative error", fontsize=14)
    ax.set_xlim(-0.9, 0.75)
    ax.set_title("With mean", fontsize=16)

    ax = fig.add_subplot(212)
    _inx0 = velRecon_chop['choprcnt'] == 0.0
    _inx1 = velRecon_chop['choprcnt'] == 1.0
    _inx5 = velRecon_chop['choprcnt'] == 5.0
    _inx10 = velRecon_chop['choprcnt'] == 10.0
    _err0 = (velRecon_chop[_inx0]['ldljra-wom-chop'] - velRecon_chop[_inx0]['ldlja']) / np.abs(velRecon_chop[_inx0]['ldlja'])
    _err1 = (velRecon_chop[_inx1]['ldljra-wom-chop'] - velRecon_chop[_inx1]['ldlja']) / np.abs(velRecon_chop[_inx1]['ldlja'])
    _err5 = (velRecon_chop[_inx5]['ldljra-wom-chop'] - velRecon_chop[_inx5]['ldlja']) / np.abs(velRecon_chop[_inx5]['ldlja'])
    _err10 = (velRecon_chop[_inx10]['ldljra-wom-chop'] - velRecon_chop[_inx10]['ldlja']) / np.abs(velRecon_chop[_inx10]['ldlja'])

    hist0, bin_edges0 = np.histogram(_err0, bins=100, density=True)
    hist1, bin_edges1 = np.histogram(_err1, bins=100, density=True)
    hist5, bin_edges5 = np.histogram(_err5, bins=100, density=True)
    hist10, bin_edges10 = np.histogram(_err10, bins=100, density=True)

    ax.plot(bin_edges0[1:], hist0, color='0.7', lw=4, label="0%")
    ax.plot(bin_edges1[1:], hist1, lw=2, label="1%")
    ax.plot(bin_edges5[1:], hist5, lw=2, label="5%")
    ax.plot(bin_edges10[1:], hist10, lw=2, label="10%")
    ax.legend(loc=2, fontsize=14)
    ax.set_xlabel("Relative error", fontsize=14)
    ax.set_xlim(-0.9, 0.75)
    ax.set_title("Without mean", fontsize=16)

    plt.tight_layout()

    return fig


def dlja_imu_factors(movement, grav, fs, scale="ap1"):
    # first enforce data into an numpy array.
    movement = np.array(movement)
    r, c = np.shape(movement)
    if r < 3:
        _str = '\n'.join(
            ("Data is too short to calcalate jerk! Data must",
             "have at least 3 samples ({0} given).".format(r)))
        raise Exception(_str)
        return

    dt = 1. / fs

    # jerk
    jerk = np.linalg.norm(np.diff(movement, axis=0, n=1), axis=1)
    jerk /= np.power(dt, 1)
    mjerk = np.sum(np.power(jerk, 2)) * dt

    # time.
    _N = len(movement)
    mdur = np.power(_N * dt, 1)

    # amplitude.
    if scale == "ap1":
        # |as^2 - g^2|
        _amp = np.abs(np.power(np.linalg.norm(movement, axis=1), 2) -
                      np.power(np.linalg.norm(grav), 2))
        mamp = np.max(_amp)
    elif scale == "ap2":
        # | as - g |^2
        _amp = np.power(np.linalg.norm(movement, axis=1) -
                        np.linalg.norm(grav), 2)
        mamp = np.max(_amp)
    elif scale == "ap3":
        # (as - mean(as))^2
        _amp = np.power(np.linalg.norm(movement - np.mean(movement, axis=0)), 2)
        mamp = np.max(_amp)

    # dlj factors
    return mdur, mamp, mjerk


def ldlja_imu_factors(movement, grav, fs, scale="ap1"):
    dljaimufac = dlja_imu_factors(movement, grav, fs, scale)
    return (- np.log(dljaimufac[0]), np.log(dljaimufac[1]),
            - np.log(dljaimufac[2]))


def ldlja_imu(movement, grav, fs, scale="ap1"):
    ldljaimufac = ldlja_imu_factors(movement, grav, fs, scale)
    return ldljaimufac[0] + ldljaimufac[1] + ldljaimufac[2]


def get_measure_data_for_perr(data, meas):
    _perr = data['phierr'].unique()
    _m = []
    for _p in _perr:
        _inx = (data['phierr'] == _p)
        _m.append(np.array(data[_inx][meas]))
    return np.array(_m)


def get_smoothness_for_via_and_dur(data, nv, nd, params):
    """Returns the smoothness values, and the relative smoothness error for
    movements with the given number of via points, and given duration.
    """
    _vinx = data['via'] == params.nVia[nv]
    _dinx = data['dur'] == params.Dur[nd]
    _vdata = data[_vinx & _dinx]
    # _perr = _vdata['phierr'].unique()
    meas = ['sparc', 'sparcr', 'sparcr-wom',
            'ldljv', 'ldljrv', 'ldljrv-wom',
            'ldlja', 'ldljra', 'ldljra-wom',
            'ldljimu', 'ldljrimu']
    _smooth = {}
    for m in meas:
        _smooth[m] = get_measure_data_for_perr(_vdata, meas=m)

    # Relative error calculation
    rerr = {}
    _v1, _v2 = _smooth['sparcr'], _smooth['sparc']
    rerr['sparcr'] = 100 * (_v1 - _v2) / np.abs(_v2)
    _v1, _v2 = _smooth['sparcr-wom'], _smooth['sparc']
    rerr['sparcr-wom'] = 100 * (_v1 - _v2) / np.abs(_v2)
    _v1, _v2 = _smooth['ldljrv'], _smooth['ldljv']
    rerr['ldljrv'] = 100 * (_v1 - _v2) / np.abs(_v2)
    _v1, _v2 = _smooth['ldljrv-wom'], _smooth['ldljv']
    rerr['ldljrv-wom'] = 100 * (_v1 - _v2) / np.abs(_v2)
    _v1, _v2 = _smooth['ldljra'], _smooth['ldlja']
    rerr['ldljra'] = 100 * (_v1 - _v2) / np.abs(_v2)
    _v1, _v2 = _smooth['ldljra-wom'], _smooth['ldlja']
    rerr['ldljra-wom'] = 100 * (_v1 - _v2) / np.abs(_v2)

    return _smooth, rerr


def get_ldlj_factors_for_via_and_dur(data, nv, nd, params):
    _vinx = data['via'] == params.nVia[nv]
    _dinx = data['dur'] == params.Dur[nd]
    _vdata = data[_vinx & _dinx]
    _perr = _vdata['phierr'].unique()
    meas = ["ldljv", "ldljrv", "ldljrv-wom",
            "ldlja", "ldljra", "ldljra-wom",
            "ldljv-T", "ldljv-A", "ldljv-J",
            "ldljrv-T", "ldljrv-A", "ldljrv-J",
            "ldljrv-wom-T", "ldljrv-wom-A", "ldljrv-wom-J",
            "ldlja-T", "ldlja-A", "ldlja-J",
            "ldljra-T", "ldljra-A", "ldljra-J",
            "ldljra-wom-T", "ldljra-wom-A", "ldljra-wom-J"]
    _smooth = {}
    for m in meas:
        _smooth[m] = get_measure_data_for_perr(_vdata, meas=m)

    return _smooth


def get_med_percentiles(data, meas, pcntl=[10, 50, 90]):
    return (np.percentile(data[meas], q=pcntl[0], axis=1),
            np.percentile(data[meas], q=pcntl[1], axis=1),
            np.percentile(data[meas], q=pcntl[2], axis=1))


def generate_figure_for_via(data, nv, params):
    fig = plt.figure(figsize=(12, 6))
    for nd, _ in enumerate(Params.Dur):
        smooth, rerr = get_smoothness_for_via_and_dur(data, nv, nd, Params)
        row, col = 3, len(Params.Dur)
        ax = fig.add_subplot(row, col, nd + 1)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _x = np.arange(1, len(Params.phiErr) + 1, 1.0)
        # SPARCr
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="SPARC")
        # SPARCr-wom
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="SPARC-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-50, 100)
        if (nd == 0):
            ax.set_ylabel("SPARC Rel. Error (%)")
        # Subplot title
        ax.set_title("Dur: {0}sec".format(params.Dur[nd]))
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, col + nd + 1)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        # LDLJ-Vr
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="SPARC")
        # LDLJ-Vr-wom
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="SPARC-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-50, 100)
        if (nd == 0):
            ax.set_ylabel("LDLJV Rel. Error (%)")
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, 2 * col + nd + 1)
        # LDLJ-Ar
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # LDLJ-Ar-wom
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-50, 100)
        ax.set_xlabel("Max. reconst. error (deg)")
        if (nd == 0):
            ax.set_ylabel("LDLJA Rel. Error (%)")
        if nd == len(params.Dur) - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
                      fancybox=True, shadow=True, ncol=1)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    _titlestr = "Relative Smoothness Error vs. Reconstruction Error(Nv: {0})"
    fig.suptitle(_titlestr.format(params.nVia[nv]), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_figure_for_via2(data, nv, params):
    fig = plt.figure(figsize=(12, 4.5))
    for nd, _ in enumerate(Params.Dur):
        smooth, rerr = get_smoothness_for_via_and_dur(data, nv, nd, Params)
        row, col = 2, len(Params.Dur)
        ax = fig.add_subplot(row, col, nd + 1)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _x = np.arange(1, len(Params.phiErr) + 1)
        # SPARC
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5)
        # LDLJV
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5)
        # LDLJA
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5)
        ax.set_xlim(1, len(Params.phiErr))
        ax.set_ylim(-50, 100)
        ax.set_xticklabels(params.phiErr)
        if (nd == 0):
            ax.set_ylabel("Rel. Error (%)")
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, col + nd + 1)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _x = np.arange(1, len(Params.phiErr) + 1)
        # SPARC
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="SPARC")
        # LDLJV
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="LDLJ-V")
        # LDLJA
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="LDLJ-A")
        ax.set_xlim(1, len(Params.phiErr))
        ax.set_xlabel("Reconst. Error (deg)")
        ax.set_ylim(-50, 100)
        ax.set_xticklabels(params.phiErr)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        if (nd == 0):
            ax.set_ylabel("Rel. Error (%)")
        if nd == len(Params.Dur) - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
                      fancybox=True, shadow=True, ncol=1)

    _titlestr = "Relative Smoothness Error vs. Reconstruction Error(Nv: {0})"
    fig.suptitle(_titlestr.format(Params.nVia[nv]), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_figure_for_via3(data, nv, sgth, params):
    fig = plt.figure(figsize=(12, 6))
    sginx = data['sgr'] >= sgth
    for nd, _ in enumerate(Params.Dur):
        _, rerr = get_smoothness_for_via_and_dur(data, nv, nd, Params)
        _, rerrsgth = get_smoothness_for_via_and_dur(data[sginx], nv, nd,
                                                     Params)
        row, col = 3, len(Params.Dur)
        # Make sure the data is not empty
        if len(rerrsgth['sparcr']) == 0:
            axbg = (0.85, 0.85, 0.85)
        else:
            axbg = (1.0, 1.0, 1.0)
        # Plot only if there is something to plot.
        ax = fig.add_subplot(row, col, nd + 1, facecolor=axbg)
        # ax.set_axis_bgcolor(axbg)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _x = np.arange(1, len(Params.phiErr) + 1)
        # SPARCr
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="SPARC")
        # SPARCr-wom
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="SPARC-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-50, 100)
        if (nd == 0):
            ax.set_ylabel("SPARC Rel. Error (%)")
        # Subplot title
        ax.set_title("Dur: {0}sec".format(params.Dur[nd]))
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, col + nd + 1, facecolor=axbg)
        # ax.set_axis_bgcolor(axbg)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        # LDLJ-Vr
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="SPARC")
        # LDLJ-Vr-wom
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="SPARC-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-50, 100)
        if (nd == 0):
            ax.set_ylabel("LDLJV Rel. Error (%)")
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, 2 * col + nd + 1, facecolor=axbg)
        # ax.set_axis_bgcolor(axbg)
        # LDLJ-Ar
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # LDLJ-Ar-wom
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-50, 100)
        ax.set_xlabel("Max. reconst. error (deg)")
        if (nd == 0):
            ax.set_ylabel("LDLJA Rel. Error (%)")
        if nd == len(params.Dur) - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
                      fancybox=True, shadow=True, ncol=1)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    _titlestr = "Relative Smoothness Error vs. Reconstruction Error(Nv: {0})"
    fig.suptitle(_titlestr.format(params.nVia[nv]), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_figure_for_via4(data, nv, sgth, params):
    fig = plt.figure(figsize=(12, 4.5))
    sginx = data['sgr'] >= sgth
    for nd, _ in enumerate(Params.Dur):
        _, rerr = get_smoothness_for_via_and_dur(data, nv, nd, Params)
        _, rerrsgth = get_smoothness_for_via_and_dur(data[sginx], nv, nd,
                                                     Params)
        row, col = 2, len(Params.Dur)
        # Make sure the data is not empty
        if len(rerrsgth['sparcr']) == 0:
            axbg = (0.85, 0.85, 0.85)
        else:
            axbg = (1.0, 1.0, 1.0)
        ax = fig.add_subplot(row, col, nd + 1, facecolor=axbg)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _x = np.arange(1, len(Params.phiErr) + 1)
        # SPARC
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5)
        # LDLJV
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5)
        # LDLJA
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5)
        ax.set_xlim(1, len(Params.phiErr))
        ax.set_ylim(-50, 100)
        ax.set_xticklabels(params.phiErr)
        if (nd == 0):
            ax.set_ylabel("Rel. Error (%)")
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, col + nd + 1, facecolor=axbg)
        plt.axhline(0, color='k', alpha=0.2, lw=1)
        _x = np.arange(1, len(Params.phiErr) + 1)
        # SPARC
        _e10, _e50, _e90 = get_med_percentiles(rerr, "sparcr-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="SPARC")
        # LDLJV
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljrv-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="LDLJ-V")
        # LDLJA
        _e10, _e50, _e90 = get_med_percentiles(rerr, "ldljra-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="LDLJ-A")
        ax.set_xlim(1, len(Params.phiErr))
        ax.set_xlabel("Reconst. Error (deg)")
        ax.set_ylim(-50, 100)
        ax.set_xticklabels(params.phiErr)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        if (nd == 0):
            ax.set_ylabel("Rel. Error (%)")
        if nd == len(Params.Dur) - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
                      fancybox=True, shadow=True, ncol=1)

    _titlestr = "Relative Smoothness Error vs. Reconstruction Error(Nv: {0})"
    fig.suptitle(_titlestr.format(Params.nVia[nv]), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_figure_for_via_for_smoothness(data, nv, params):
    fig = plt.figure(figsize=(15, 10))
    for nd, _ in enumerate(Params.Dur):
        smooth, rerr = get_smoothness_for_via_and_dur(data, nv, nd, Params)
        row, col = 4, len(Params.Dur)
        # print row, col
        ax = fig.add_subplot(row, col, nd + 1)
        _x = np.arange(1, len(Params.phiErr) + 1)
        # SPARC
        _e10, _e50, _e90 = get_med_percentiles(smooth, "sparc")
        # print _e10, _e50, _e90
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="O")
        # SPARCr
        _e10, _e50, _e90 = get_med_percentiles(smooth, "sparcr")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # SPARCr-wom
        _e10, _e50, _e90 = get_med_percentiles(smooth, "sparcr-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-3, -1)
        if (nd == 0):
            ax.set_ylabel("SPARC")
        # Subplot title
        ax.set_title("Dur: {0}sec".format(params.Dur[nd]))
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, col + nd + 1)
        # plt.axhline(0, color='k', alpha=0.2, lw=1)
        # LDLJ-V
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="O")
        # LDLJ-Vr
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljrv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # LDLJ-Vr-wom
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljrv-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-13, 6)
        if (nd == 0):
            ax.set_ylabel("LDLJV")
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, 2 * col + nd + 1)
        # LDLJA
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldlja")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="O")
        # LDLJ-Ar
        # ax.plot([0, 7], [0, 0], 'k', alpha=0.2, lw=1)
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljra")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # LDLJ-Ar-wom
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljra-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-7, 6)
        ax.set_xlabel("Max. reconst. error (deg)")
        if (nd == 0):
            ax.set_ylabel("LDLJAs")
        if nd == len(params.Dur) - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
                      fancybox=True, shadow=True, ncol=1)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, 3 * col + nd + 1)
        # LDLJA
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldlja")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="O")
        # LDLJ-AIMU
        # ax.plot([0, 7], [0, 0], 'k', alpha=0.2, lw=1)
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljimu")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # # LDLJ-Ar-wom
        # _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljra-wom")
        # ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
        #                 alpha=0.1)
        # ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
        #         label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-7, 6)
        ax.set_xlabel("Max. reconst. error (deg)")
        if (nd == 0):
            ax.set_ylabel("LDLJ IMU")
        if nd == len(params.Dur) - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
                      fancybox=True, shadow=True, ncol=1)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

    _titlestr = "Smoothness Value vs. Reconstruction Error(Nv: {0})"
    fig.suptitle(_titlestr.format(params.nVia[nv]), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_figure_for_via_for_ldlj_details(data, nv, params):
    fig = plt.figure(figsize=(15, 7.5))
    for nd, _ in enumerate(Params.Dur):
        smooth = get_ldlj_factors_for_via_and_dur(data, nv, nd, Params)
        row, col = 4, len(Params.Dur)
        ax = fig.add_subplot(row, col, nd + 1)
        _x = np.arange(1, len(Params.phiErr) + 1)
        # LDLJ-V
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="O")
        # LDLJ-Vr
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljrv")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # LDLJ-Vr-wom
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljrv-wom")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        # ax.set_ylim(-3, -1)
        if (nd == 0):
            ax.set_ylabel("LDLJV")
        # Subplot title
        ax.set_title("Dur: {0}sec".format(params.Dur[nd]))
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, col + nd + 1)
        # LDLJ-V-T
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljv-T")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="O")
        # LDLJ-V-A
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljv-A")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="R")
        # LDLJ-V-J
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljv-J")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="R-wom")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        ax.set_ylim(-16, 10)
        if (nd == 0):
            ax.set_ylabel("LDLJV Factors")
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, 2 * col + nd + 1)
        # LDLJRV-T
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljrv-T")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="T")
        # LDLJRV-A
        # ax.plot([0, 7], [0, 0], 'k', alpha=0.2, lw=1)
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljrv-A")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="A")
        # LDLJRV-J
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljrv-J")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="J")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        # ax.set_ylim(-7, 6)
        ax.set_xlabel("Max. reconst. error (deg)")
        if (nd == 0):
            ax.set_ylabel("LDLJRV Factors")
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(row, col, 3 * col + nd + 1)
        # LDLJRV-T
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljra-T")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["medium green"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["medium green"], alpha=0.5,
                label="T")
        # LDLJRV-A
        # ax.plot([0, 7], [0, 0], 'k', alpha=0.2, lw=1)
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljra-A")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["pale red"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["pale red"], alpha=0.5,
                label="A")
        # LDLJRV-J
        _e10, _e50, _e90 = get_med_percentiles(smooth, "ldljra-J")
        ax.fill_between(_x, _e10, _e90, facecolor=sb.xkcd_rgb["denim blue"],
                        alpha=0.1)
        ax.plot(_x, _e50, color=sb.xkcd_rgb["denim blue"], alpha=0.5,
                label="J")
        ax.set_xlim(1, len(params.phiErr))
        ax.set_xticks(_x)
        ax.set_xticklabels(params.phiErr)
        # ax.set_ylim(-7, 6)
        ax.set_xlabel("Max. reconst. error (deg)")
        if (nd == 0):
            ax.set_ylabel("LDLJRA Factors")
        if nd == len(params.Dur) - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.0),
                      fancybox=True, shadow=True, ncol=1)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    _titlestr = "LDLJ Factors vs. Reconstruction Error(Nv: {0})"
    fig.suptitle(_titlestr.format(params.nVia[nv]), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_smooth_compare_scatter(data, cols, lims):
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(121)
    ax.plot(lims, lims, 'k', alpha=0.2)
    ax.plot(data[cols[0]], data[cols[1]], '.', alpha=0.05)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconst.")
    ax.set_title("Reconstructed vs. Original")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax = fig.add_subplot(122)
    ax.plot(lims, lims, 'k', alpha=0.2)
    ax.plot(data[cols[0]], data[cols[2]], '.', alpha=0.05)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_title("Reconstructed WOM vs. Original")
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconst. WOM")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig


def generate_smooth_compare_scatter_sgrth(data, cols, lims, sgrth):
    sgrinx = data['sgr'] > sgrth
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(121)
    ax.plot(lims, lims, 'k', alpha=0.2)
    ax.plot(data[cols[0]], data[cols[1]], '.', alpha=0.05)
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[1]], '+', alpha=0.3)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconst.")
    ax.set_title("Reconstructed vs. Original")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax = fig.add_subplot(122)
    ax.plot(lims, lims, 'k', alpha=0.2)
    ax.plot(data[cols[0]], data[cols[2]], '.', alpha=0.05)
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[2]], '+', alpha=0.3)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_title("Reconstructed WOM vs. Original")
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconst. WOM")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig


def generate_smootherr_vs_sigmeas(data, cols, lims, meas):
    _rerr = 100 * (data[cols[1]] - data[cols[0]]) / np.abs(data[cols[0]])
    _rerrwom = 100 * (data[cols[2]] - data[cols[0]]) / np.abs(data[cols[0]])
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(121)
    ax.plot(data[meas], _rerr, '.', alpha=0.05)
    ax.set_ylim(*lims)
    ax.set_xlabel(meas)
    ax.set_ylabel("Rel Err. (%)")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax = fig.add_subplot(122)
    ax.plot(data[meas], _rerrwom, '.', alpha=0.05)
    ax.set_ylim(*lims)
    ax.set_xlabel(meas)
    ax.set_ylabel("Rel Err. WOM (%)")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig


def generate_smooth_compare_scatter_diff_sgr(data, cols, lims):
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(121)
    ax.plot(lims, lims, 'k', alpha=0.2)
    # Between 1 and 1.5
    sgrinx = ((data['sgr'] >= 1.0) &
              (data['sgr'] < 1.5))
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[1]], 'r.', alpha=0.05)
    # Between 1.5 and 2.0
    sgrinx = ((data['sgr'] >= 1.5) &
              (data['sgr'] < 2.0))
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[1]], 'go', alpha=0.1)
    # Above 2.0
    sgrinx = (data['sgr'] >= 2.0)
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[1]], 'b+', alpha=0.5)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconst.")
    ax.set_title("Reconstructed vs. Original")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax = fig.add_subplot(122)
    ax.plot(lims, lims, 'k', alpha=0.2)
    # Between 1 and 1.5
    sgrinx = ((data['sgr'] >= 1.0) &
              (data['sgr'] < 1.5))
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[2]], 'r.', alpha=0.05)
    # Between 1.5 and 2.0
    sgrinx = ((data['sgr'] >= 1.5) &
              (data['sgr'] < 2.0))
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[2]], 'go', alpha=0.1)
    # Above 2.0
    sgrinx = (data['sgr'] >= 2.0)
    ax.plot(data[sgrinx][cols[0]], data[sgrinx][cols[2]], 'b+', alpha=0.5)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_title("Reconstructed WOM vs. Original")
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconst. WOM")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig


def generate_smoothness_summary_plots(velRecon, params):
    # Smoothness values for different via points, durations and reconst. errors.
    sys.stdout.write("Smoothness summary ... ")
    _fname = "{0}/summary/smoothval/smoothval_r_vs_rwom.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothval/smoothval_r_vs_rwom_{1:02d}.{2}"
    with PdfPages(_fname) as pdf:
        for nv, _ in enumerate(params.nVia):
            fig = generate_figure_for_via_for_smoothness(velRecon, nv, params);
            # Summary PDF
            pdf.savefig(fig)
            # Individual SVG and PNG files
            fig.savefig(_fnamestr.format(params.outdir, nv, "png"), format="png", dpi=600)
            fig.savefig(_fnamestr.format(params.outdir, nv, "svg"), format="svg")
            plt.close()
    sys.stdout.write("Done \n")

    # Smoothness LDLJ factors.
    sys.stdout.write("Smoothness LDLJ factors summary ... ")
    _fname = "{0}/summary/smoothval/ldljfac_r_vs_rwom.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothval/ldljfac_r_vs_rwom_{1:02d}.{2}"
    with PdfPages(_fname) as pdf:
        for nv, _ in enumerate(params.nVia):
            fig = generate_figure_for_via_for_ldlj_details(velRecon, nv, params);
            # Summary PDF
            pdf.savefig(fig)
            # Individual SVG and PNG files
            fig.savefig(_fnamestr.format(params.outdir, nv, "png"), format="png", dpi=600)
            fig.savefig(_fnamestr.format(params.outdir, nv, "svg"), format="svg")
            plt.close()
    sys.stdout.write("Done \n")

    # Relative smoothness error
    sys.stdout.write("Relative smoothness error summary ... ")
    _fname = "{0}/summary/relerr/relerr_r_vs_rwom.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/relerr/relerr_r_vs_rwom_{1:02d}.{2}"
    with PdfPages(_fname) as pdf:
        for nv, _ in enumerate(params.nVia):
            fig = generate_figure_for_via(velRecon, nv, params);
            # Summary PDF
            pdf.savefig(fig)
            # Individual SVG and PNG files
            fig.savefig(_fnamestr.format(params.outdir, nv, "png"), format="png", dpi=600)
            fig.savefig(_fnamestr.format(params.outdir, nv, "svg"), format="svg")
            plt.close()
    sys.stdout.write("Done \n")

    # Relative smoothness error for different SGR
    sys.stdout.write("Relative smoothness error (different SGR) summary ... ")
    _fname = "{0}/summary/relerr/relerr_r_vs_rwom_high_sgr.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/relerr/relerr_r_vs_rwom_high_sgr_{1:02d}.{2}"
    with PdfPages(_fname) as pdf:
        for nv, _ in enumerate(params.nVia):
            fig = generate_figure_for_via3(velRecon, nv, 2.0, params);
            # Summary PDF
            pdf.savefig(fig)
            # Individual SVG and PNG files
            fig.savefig(_fnamestr.format(params.outdir, nv, "png"), format="png", dpi=600)
            fig.savefig(_fnamestr.format(params.outdir, nv, "svg"), format="svg")
            plt.close()
    sys.stdout.write("Done \n")

    # Relative smoothness error comparisons
    sys.stdout.write("Relative smoothness error comparisons ... ")
    _fname = "{0}/summary/relerr/relerr_diff_smooth.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/relerr/relerr_diff_smooth_{1:02d}.{2}"
    with PdfPages(_fname) as pdf:
        for nv, _ in enumerate(params.nVia):
            fig = generate_figure_for_via2(velRecon, nv, params);
            pdf.savefig(fig)
            # Individual SVG and PNG files
            fig.savefig(_fnamestr.format(params.outdir, nv, "png"), format="png", dpi=600)
            fig.savefig(_fnamestr.format(params.outdir, nv, "svg"), format="svg")
            plt.close()
    sys.stdout.write("Done \n")

    # Relative smoothness error comparisons
    sys.stdout.write("Relative smoothness error comparisons (different SGR) ... ")
    _fname = "{0}/summary/relerr/relerr_diff_smooth_high_sgr.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/relerr/relerr_diff_smooth_high_sgr_{1:02d}.{2}"
    with PdfPages(_fname) as pdf:
        for nv, _ in enumerate(params.nVia):
            fig = generate_figure_for_via4(velRecon, nv, 2.0, params);
            # Summary PDF
            pdf.savefig(fig)
            # Individual SVG and PNG files
            fig.savefig(_fnamestr.format(params.outdir, nv, "png"), format="png", dpi=600)
            fig.savefig(_fnamestr.format(params.outdir, nv, "svg"), format="svg")
            plt.close()
    sys.stdout.write("Done \n")


def generate_smoothness_scatter_plots(velRecon, params):
    # Original versus reconst. smoothness with and without mean.
    _fname = "{0}/summary/smoothscatter/smooth_scatter.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothscatter/smooth_scatter_{1}.{2}"
    sys.stdout.write(_fname + "...")
    with PdfPages(_fname) as pdf:
        # SPARC
        fig = generate_smooth_compare_scatter(velRecon, cols=["sparc", "sparcr", "sparcr-wom"],
                                              lims=[-3, -1.25]);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "svg"), format="svg")
        plt.close()

        # LDLJ-V
        fig = generate_smooth_compare_scatter(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                              lims=[-12.5, 0.0]);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "svg"), format="svg")
        plt.close()

        # LDLJ-A
        fig = generate_smooth_compare_scatter(velRecon, cols=["ldlja", "ldljra", "ldljra-wom"],
                                              lims=[-6.0, 0.0]);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "svg"), format="svg")
        plt.close()
    sys.stdout.write(" Done!\n")

    # Smoothness vs. SGR
    _fname = "{0}/summary/smoothscatter/smootherr_vs_sgr.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothscatter/smootherr_vs_sgr_{1}.{2}"
    sys.stdout.write(_fname + "...")
    with PdfPages(_fname) as pdf:
        # SPARC
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["sparc", "sparcr", "sparcr-wom"],
                                            lims=[-40, 100], meas="sgr");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "svg"), format="svg")
        plt.close()

        # LDLJ-V
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                            lims=[-40, 100], meas="sgr");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "svg"), format="svg")
        plt.close()

        # LDLJ-A
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                            lims=[-40, 100], meas="sgr");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "svg"), format="svg")
        plt.close()
    sys.stdout.write(" Done!\n")

    # Smoothness vs. angle arange
    _fname = "{0}/summary/smoothscatter/smootherr_vs_angrange.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothscatter/smootherr_vs_angrange_{1}.{2}"
    sys.stdout.write(_fname + "...")
    with PdfPages(_fname) as pdf:
        # SPARC
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["sparc", "sparcr", "sparcr-wom"],
                                            lims=[-40, 100], meas="angrange");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "svg"), format="svg")
        plt.close()

        # LDLJ-V
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                            lims=[-40, 100], meas="angrange");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "svg"), format="svg")
        plt.close()

        # LDLJ-A
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                            lims=[-40, 100], meas="angrange");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "svg"), format="svg")
        plt.close()
    sys.stdout.write(" Done!\n")

    # Smoothness vs. max. angular velocity
    _fname = "{0}/summary/smoothscatter/smootherr_vs_angvelmax.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothscatter/smootherr_vs_angvelmax_{1}.{2}"
    sys.stdout.write(_fname + "...")
    with PdfPages(_fname) as pdf:
        # SPARC
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["sparc", "sparcr", "sparcr-wom"],
                                            lims=[-40, 100], meas="angvelmax");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "svg"), format="svg")
        plt.close()

        # LDLJ-V
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                            lims=[-40, 100], meas="angvelmax");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "svg"), format="svg")
        plt.close()

        # LDLJ-A
        fig = generate_smootherr_vs_sigmeas(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                            lims=[-40, 100], meas="angvelmax");
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "svg"), format="svg")
        plt.close()
    sys.stdout.write(" Done!\n")

    # Original vs. reconst. smoothness for differet SGR
    _fname = "{0}/summary/smoothscatter/smooth_scatter_above_sgr.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothscatter/smooth_scatter_above_sgr_{1}.{2}"
    sys.stdout.write(_fname + "...")
    with PdfPages(_fname) as pdf:
        # SPARC
        fig = generate_smooth_compare_scatter_sgrth(velRecon, cols=["sparc", "sparcr", "sparcr-wom"],
                                                    lims=[-3, -1.25], sgrth=2.0);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "svg"), format="svg")
        plt.close()

        # LDLJ-V
        fig = generate_smooth_compare_scatter_sgrth(velRecon, cols=["ldljv", "ldljrv", "ldljrv-wom"],
                                                    lims=[-12.5, 0.0], sgrth=2.0);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "svg"), format="svg")
        plt.close()

        # LDLJ-A
        fig = generate_smooth_compare_scatter_sgrth(velRecon, cols=["ldlja", "ldljra", "ldljra-wom"],
                                                    lims=[-6.0, 0.0], sgrth=2.0);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "svg"), format="svg")
        plt.close()
    sys.stdout.write(" Done!\n")

    _fname = "{0}/summary/smoothscatter/smooth_scatter_diff_sgr.pdf".format(params.outdir)
    _fnamestr = "{0}/summary/smoothscatter/smooth_scatter_diff_sgr_{1}.{2}"
    sys.stdout.write(_fname + "...")
    with PdfPages(_fname) as pdf:
        # SPARC
        fig = generate_smooth_compare_scatter_diff_sgr(velRecon, ["sparc", "sparcr", "sparcr-wom"],
                                                          [-3, -1.25]);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "sparc", "svg"), format="svg")
        plt.close()

        # LDLJ-V
        fig = generate_smooth_compare_scatter_diff_sgr(velRecon, ["ldljv", "ldljrv", "ldljrv-wom"],
                                                          [-12.5, 0.0]);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldljv", "svg"), format="svg")
        plt.close()

        # LDLJ-A
        fig = generate_smooth_compare_scatter_diff_sgr(velRecon, ["ldlja", "ldljra", "ldljra-wom"],
                                                          [-6.0, 0.0]);
        pdf.savefig(fig)
        # Individual SVG and PNG files
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "png"), format="png", dpi=600)
        fig.savefig(_fnamestr.format(params.outdir, "ldlja", "svg"), format="svg")
        plt.close()
    sys.stdout.write(" Done!\n")


def generate_smooth_compare_scatter_for_paper(data, lims, sgrth):
    fig = plt.figure(figsize=(12, 5.5))
    # SPARC
    ax = fig.add_subplot(241)
    ax.plot(lims[0], lims[0], 'k', alpha=0.2)
    _r = stats.pearsonr(data["sparc"], data["sparcr"])[0]
    ax.plot(data["sparc"], data["sparcr"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r))
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[0])
    ax.set_ylabel("Reconst.")
    ax.set_title("SPARC")
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    
    # LDLJ-V
    ax = fig.add_subplot(242)
    _r = stats.pearsonr(data["ldljv"], data["ldljrv"])[0]
    ax.plot(lims[1], lims[1], 'k', alpha=0.2)
    ax.plot(data["ldljv"], data["ldljrv"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r))
    ax.set_xlim(*lims[1])
    ax.set_ylim(*lims[1])
    ax.set_title("LDLJV")
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    
    # LDLJ-A
    ax = fig.add_subplot(243)
    _r = stats.pearsonr(data["ldlja"], data["ldljra"])[0]
    ax.plot(lims[2], lims[2], 'k', alpha=0.2)
    ax.plot(data["ldlja"], data["ldljra"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r))
    ax.set_xlim(*lims[2])
    ax.set_ylim(*lims[2])
    ax.set_title("LDLJA")
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)

    # LDLJ-A1
    sgrinx = data['sgr'] > sgrth
    ax = fig.add_subplot(244)
    _r1 = stats.pearsonr(data["ldlja"], data["ldlja1"])[0]
    _r2 = stats.pearsonr(data[sgrinx]["ldlja"], data[sgrinx]["ldlja1"])[0]
    ax.plot(lims[3], lims[3], 'k', alpha=0.2)
    ax.plot(data["ldlja"], data["ldlja1"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r1))
    ax.plot(data[sgrinx]['ldlja'], data[sgrinx]['ldlja1'], '+', alpha=0.2, label="$r = {0:0.3f}$".format(_r2))
    ax.set_xlim(*lims[3])
    ax.set_ylim(*lims[3])
    ax.set_title("LDLJA1 / LDLJA2")
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    
    # SPARCR
    ax = fig.add_subplot(245)
    _r1 = stats.pearsonr(data["sparc"], data["sparcr-wom"])[0]
    _r2 = stats.pearsonr(data[sgrinx]["sparc"], data[sgrinx]["sparcr-wom"])[0]
    ax.plot(lims[0], lims[0], 'k', alpha=0.2)
    ax.plot(data["sparc"], data["sparcr-wom"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r1))
    ax.plot(data[sgrinx]['sparc'], data[sgrinx]['sparcr-wom'], '+', alpha=0.2, label="$r = {0:0.3f}$".format(_r2))
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[0])
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconst.")
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    
    # LDLJV-R
    ax = fig.add_subplot(246)
    _r1 = stats.pearsonr(data["ldljv"], data["ldljrv-wom"])[0]
    _r2 = stats.pearsonr(data[sgrinx]["ldljv"], data[sgrinx]["ldljrv-wom"])[0]
    ax.plot(lims[1], lims[1], 'k', alpha=0.2)
    ax.plot(data["ldljv"], data["ldljrv-wom"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r1))
    ax.plot(data[sgrinx]['ldljv'], data[sgrinx]['ldljrv-wom'], '+', alpha=0.2, label="$r = {0:0.3f}$".format(_r2))
    ax.set_xlim(*lims[1])
    ax.set_ylim(*lims[1])
    ax.set_xlabel("Original")
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    
    # LDLJA-R
    ax = fig.add_subplot(247)
    _r1 = stats.pearsonr(data["ldlja"], data["ldljra-wom"])[0]
    _r2 = stats.pearsonr(data[sgrinx]["ldlja"], data[sgrinx]["ldljra-wom"])[0]
    _r = stats.pearsonr(data["ldlja"], data["ldljra-wom"])[0]
    ax.plot(lims[2], lims[2], 'k', alpha=0.2)
    ax.plot(data["ldlja"], data["ldljra-wom"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r1))
    ax.plot(data[sgrinx]['ldlja'], data[sgrinx]['ldljra-wom'], '+', alpha=0.2, label="$r = {0:0.3f}$".format(_r2))
    ax.set_xlim(*lims[2])
    ax.set_ylim(*lims[2])
    ax.set_xlabel("Original")
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.legend(loc=4, frameon=False, fontsize=12)

    # LDLJ-A2
    ax = fig.add_subplot(248)
    _r1 = stats.pearsonr(data["ldlja"], data["ldlja2"])[0]
    _r2 = stats.pearsonr(data[sgrinx]["ldlja"], data[sgrinx]["ldlja2"])[0]
    ax.plot(lims[4], lims[4], 'k', alpha=0.2)
    ax.plot(data["ldlja"], data["ldlja2"], '.', alpha=0.05, label="$r = {0:0.3f}$".format(_r1))
    ax.plot(data[sgrinx]['ldlja'], data[sgrinx]['ldlja2'], '+', alpha=0.2, label="$r = {0:0.3f}$".format(_r2))
    ax.set_xlim(*lims[4])
    ax.set_ylim(*lims[4])
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    return fig


def gen_summary_for_ldlja(data, sgrth, col1="ldlja", col2="ldljra",
                          lims=[-6.5,-1]):
    fig = plt.figure(figsize=(16, 3.5))
    # LDLJA (Reconstructed vs. Original)
    ax = fig.add_subplot(141)
    sgrinx = data['sgr'] >= sgrth
    notsgrinx = data['sgr'] < sgrth
    _r1 = stats.pearsonr(data[col1], data[col2])[0]
    _r2 = stats.pearsonr(data[sgrinx][col1], data[sgrinx][col2])[0]
    # lims = [-6.5, -1]
    ax.plot(lims, lims, 'k', lw=3, alpha=0.2)
    ax.plot(data[col1], data[col2], '.', alpha=0.3, label="$\\rho = {0:0.3f}$".format(_r1))
    ax.plot(data[sgrinx][col1], data[sgrinx][col2], '+', alpha=0.8, label="$\\rho = {0:0.3f}$".format(_r2))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Original", fontsize=14)
    ax.set_title("Original vs. Reconst.", fontsize=14)
    ax.set_ylabel("Reconstructed", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=2, frameon=False, fontsize=14)

    # LDLJA error historgams
    _err1 = (data[notsgrinx][col2] - data[notsgrinx][col1]) / np.abs(data[notsgrinx][col1])
    _err2 = (data[sgrinx][col2] - data[sgrinx][col1]) / np.abs(data[sgrinx][col1])
    hist1, bin_edges1 = np.histogram(_err1, bins=100, density=True)
    hist2, bin_edges2 = np.histogram(_err2, bins=50, density=True)
    ax = fig.add_subplot(142)
    ax.plot(bin_edges1[1:], hist1, lw=2.0, alpha=0.8, label=f"$SGR < {sgrth}$")
    ax.plot(bin_edges2[1:], hist2, lw=2.0, alpha=0.8, label=f"$SGR \geq {sgrth}$")
    ax.legend(loc=2, frameon=False, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Rel. error", fontsize=14)
    ax.set_title("Rel. err. historgram", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.9, 0.5)

    # LDLJA correlation vs. movement duration
    ax = fig.add_subplot(143)
    perr = data['phierr'].unique()
    dur = data['dur'].unique()
    _cols = [[col1, col2]]
    _r2vals = [np.zeros((len(perr), len(dur)))]
    for i, _p in enumerate(perr):
        _pinx = data['phierr'] == _p
        for j, _d in enumerate(dur):
            _inx = (data['dur'] == _d) & _pinx
            for k in range(len(_cols)):
                _r2vals[k][i, j] = stats.pearsonr(data[_cols[k][0]][_inx],
                                                  data[_cols[k][1]][_inx])[0]
    
    _xvals = np.arange(1, len(dur)+1)
    ax.plot(dur, _r2vals[0][0, :], color="k", lw=2, alpha=0.75,
            label="{0:0.0f} $\deg$".format(perr[0]))
    ax.plot(dur, _r2vals[0][1, :], color="k", lw=2, ls="--", alpha=0.75,
            label="{0:0.0f} $\deg$".format(perr[1]))
    ax.plot(dur, _r2vals[0][2, :], color="k", lw=2, ls=":", alpha=0.75,
            label="{0:0.0f} $\deg$".format(perr[2]))
    # ax.plot(_xvals, _r2vals[0][3, :], lw=3, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[3]))
    ax.set_xlim(2.5, len(dur))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(dur)
    ax.set_xticklabels(dur);
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_ylabel("Correlation", fontsize=14)
    ax.set_xlabel("Duration (sec)", fontsize=14)
    ax.set_title("Corr. vs. Dur.", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.legend(loc=3, prop={'size': 14}, frameon=False)

    # Correlaion versus SGR
    ax = fig.add_subplot(144)
    _sgrth = [1.0, 1.05, 1.1, 1.2, 1.5, 1.75, 2.0, 2.5]
    _cols = [[col1, col2]]
    _r2vals = [np.zeros(len(_sgrth))]
    _revals = [np.zeros(len(_sgrth))]
    for i, _s in enumerate(_sgrth):
        _sgrinx = data['sgr'] >= _s
        for k in range(len(_cols)):
            # orrelation coefficient
            _r2vals[k][i] = stats.pearsonr(data[_cols[k][0]][_sgrinx],
                                           data[_cols[k][1]][_sgrinx])[0]
            # Relative error
            __rerr = (data[_cols[k][1]][_sgrinx] - data[_cols[k][0]][_sgrinx]) / np.abs(data[_cols[k][0]][_sgrinx])
            _revals[k][i] = np.percentile(__rerr, 95)
    _xvals = np.arange(1, len(_sgrth)+1)
    ax.plot(_sgrth, _r2vals[0], color="#d9544d", lw=2, alpha=1.0, label="Corelation")
    ax.plot(_sgrth, _revals[0], color="#39ad48", lw=2, alpha=1.0, label="Rel. error")
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0.95, 2.55)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("SGR", fontsize=14)
    ax.set_title("Corr., Rel. err. vs. SGR", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.legend(prop={'size': 14}, frameon=False)

    plt.tight_layout(pad=0.0, w_pad=-0.5, h_pad=1.0)

    return fig


def gen_summary_for_gyro(data):
    fig = plt.figure(figsize=(8, 3.5))
    # SPARC
    ax = fig.add_subplot(121)
    _r1 = stats.pearsonr(data["sparc-gyro"], data["sparcr-gyro"])[0]
    lims = [-17, 0.0]
    ax.plot(lims, lims, '0.5', alpha=0.2)
    ax.plot(data['sparc-gyro'], data['sparcr-gyro'], 'o', color=(0, 0, 0.9), alpha=0.1,
            label="$\\rho\,(SPARC) = {0:0.2f}$".format(_r1))
    ax.set_xlabel("Original", fontsize=14)
    ax.set_title("SPARC & LDLJV", fontsize=14)
    ax.set_ylabel("Reconstructed", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # LDLJV
    _r1 = stats.pearsonr(data["ldljv-gyro"], data["ldljvr-gyro"])[0]
    ax.plot(data['ldljv-gyro'], data['ldljvr-gyro'], '.', color=(0.9, 0, 0), alpha=0.1,
            label="$\\rho\,(LDLJV) = {0:0.2f}$".format(_r1))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=2, frameon=False, fontsize=14)

    # LDLJ - Rel. Error
    ax = fig.add_subplot(122)
    _err = (data["ldljvr-gyro"] - data["ldljv-gyro"]) / np.abs(data["ldljv-gyro"])
    hist, bin_edges = np.histogram(_err, bins=200, range=[-1.1, 0.2], density=True)
    ax.step(bin_edges[1:], hist, color=(0.9, 0, 0), lw=1.5, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Rel. error", fontsize=14)
    ax.set_title("Rel. err. historgram (LDLJV)", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.set_xlim(-1.1, 0.1)

    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=1.0)

    return fig


def gen_summary_for_ldlja_agr(data, lims, agrth):
    fig = plt.figure(figsize=(8, 6.5))
    # LDLJA (Reconstructed vs. Original)
    ax = fig.add_subplot(221)
    sgrinx = data['agr'] > agrth
    notsgrinx = data['agr'] <= agrth
    _r1 = stats.pearsonr(data["ldlja"], data["ldljra"])[0]
    _r2 = stats.pearsonr(data[sgrinx]["ldlja"], data[sgrinx]["ldljra"])[0]
    ax.plot(lims, lims, 'k', alpha=0.2)
    ax.plot(data["ldlja"], data["ldljra"], '.', alpha=0.05, label="$\\rho = {0:0.3f}$".format(_r1))
    ax.plot(data[sgrinx]['ldlja'], data[sgrinx]['ldljra'], '+', alpha=0.2, label="$\\rho = {0:0.3f}$".format(_r2))
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Original", fontsize=14)
    ax.set_title("Original vs. Reconst.", fontsize=14)
    ax.set_ylabel("Reconstructed", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=4, frameon=False, fontsize=14)

    # LDLJA error historgams
    _err1 = (data[notsgrinx]["ldljra"] - data[notsgrinx]["ldlja"]) / np.abs(data[notsgrinx]["ldlja"])
    _err2 = (data[sgrinx]["ldljra"] - data[sgrinx]["ldlja"]) / np.abs(data[sgrinx]["ldlja"])
    hist1, bin_edges1 = np.histogram(_err1, bins=100, density=True)
    hist2, bin_edges2 = np.histogram(_err2, bins=50, density=True)
    ax = fig.add_subplot(222)
    ax.plot(bin_edges1[1:], hist1, lw=2.0, alpha=0.8, label=f"$1 \leq AGR \leq {agrth}$")
    ax.plot(bin_edges2[1:], hist2, lw=2.0, alpha=0.8, label=f"${agrth} < AGR$")
    ax.legend(loc=1, frameon=False, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Rel. error", fontsize=14)
    ax.set_title("Rel err historgram", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    # # Inset figure
    # axins = inset_axes(ax, width="65%", height="45%", loc=1)
    # axins.tick_params(axis='y',          # changes apply to the x-axis
    #                   which='both',      # both major and minor ticks are affected
    #                   left=False,      # ticks along the bottom edge are off
    #                   right=False,         # ticks along the top edge are off
    #                   labelleft=False) # labels along the bottom edge are off
    # axins.plot(bin_edges[1:], -hist, lw=1.5, alpha=0.8)
    # axins.plot(bin_edges_sgr[1:], hist_sgr, lw=1.5, alpha=0.8)
    # axins.set_xlim(-0.01, 0.05)
    # axins.grid(color='0.7', linestyle='--', linewidth=0.5)

    # LDLJA correlation vs. movement duration
    ax = fig.add_subplot(223)
    perr = data['phierr'].unique()
    dur = data['dur'].unique()
    _cols = [['ldlja', 'ldljra']]
    _r2vals = [np.zeros((len(perr), len(dur)))]
    for i, _p in enumerate(perr):
        _pinx = data['phierr'] == _p
        for j, _d in enumerate(dur):
            _inx = (data['dur'] == _d) & _pinx
            for k in range(len(_cols)):
                _r2vals[k][i, j] = stats.pearsonr(data[_cols[k][0]][_inx],
                                                  data[_cols[k][1]][_inx])[0]
    
    _xvals = np.arange(1, len(dur)+1)
    ax.plot(dur, _r2vals[0][0, :], lw=2, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[0]))
    ax.plot(dur, _r2vals[0][1, :], lw=2, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[1]))
    ax.plot(dur, _r2vals[0][2, :], lw=2, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[2]))
    # ax.plot(_xvals, _r2vals[0][3, :], lw=3, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[3]))
    ax.set_xlim(1, len(dur))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(dur)
    ax.set_xticklabels(dur);
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.set_xlabel("Duration (sec)", fontsize=14)
    ax.set_title("Corr vs. Dur", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.legend(prop={'size': 14}, frameon=False)

    # Correlaion versus SGR
    ax = fig.add_subplot(224)
    # _agrth = [1.0, 1.05, 1.1, 1.2, 1.5, 1.75, 2.0, 2.5]
    _agrth = [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    _cols = [['ldlja', 'ldljra']]
    _r2vals = [np.zeros(len(_agrth))]
    _revals = [np.zeros(len(_agrth))]
    for i, _s in enumerate(_agrth):
        _agrinx = data['agr'] > _s
        for k in range(len(_cols)):
            # orrelation coefficient
            _r2vals[k][i] = stats.pearsonr(data[_cols[k][0]][_agrinx],
                                           data[_cols[k][1]][_agrinx])[0]
            # Relative errors
            __rerr = (data[_cols[k][1]][_agrinx] - data[_cols[k][0]][_agrinx]) / np.abs(data[_cols[k][0]][_agrinx])
            _revals[k][i] = np.percentile(__rerr, 95)
    _xvals = np.arange(1, len(_agrth)+1)
    ax.plot(_agrth, _r2vals[0], lw=2, alpha=0.8, label="Corelation")
    ax.plot(_agrth, _revals[0], lw=2, alpha=0.8, label="Rel. error")
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("AGR", fontsize=14)
    ax.set_title("Corr, Rel err vs. AGR", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.legend(prop={'size': 14}, frameon=False)

    plt.tight_layout()

    return fig


def gen_correlation_summary(data):
    perr = data['phierr'].unique()
    dur = data['dur'].unique()
    _cols = [['sparc', 'sparcr'],
              ['ldljv', 'ldljrv'],
              ['ldlja', 'ldljra'],
              ['ldljv', 'ldljrv-wom'],
              ['ldlja', 'ldljra-wom']]
    _r2vals = [np.zeros((len(perr), len(dur))),
              np.zeros((len(perr), len(dur))),
              np.zeros((len(perr), len(dur))),
              np.zeros((len(perr), len(dur))),
              np.zeros((len(perr), len(dur)))]
    for i, _p in enumerate(perr):
        _pinx = data['phierr'] == _p
        for j, _d in enumerate(dur):
            _inx = (data['dur'] == _d) & _pinx
            for k in range(len(_cols)):
                _r2vals[k][i, j] = stats.pearsonr(data[_cols[k][0]][_inx],
                                                  data[_cols[k][1]][_inx])[0]
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(231)
    _xvals = np.arange(1, len(dur)+1)
    ax.plot(_xvals, _r2vals[0][0, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[0][1, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[0][2, :], lw=3, alpha=0.5)
    # ax.plot(_xvals, _r2vals[0][3, :], lw=3, alpha=0.5)
    ax.set_xlim(1, len(dur))
    ax.set_xticks(_xvals)
    ax.set_xticklabels(dur);
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.set_title("SPARC", fontsize=14)

    ax = fig.add_subplot(232)
    ax.plot(_xvals, _r2vals[1][0, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[1][1, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[1][2, :], lw=3, alpha=0.5)
    # ax.plot(_xvals, _r2vals[1][3, :], lw=3, alpha=0.5)
    ax.set_xlim(1, len(dur))
    ax.set_xticks(_xvals)
    ax.set_xticklabels(dur);
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Duration (sec)", fontsize=14)
    ax.set_title("LDLJV", fontsize=14)

    ax = fig.add_subplot(233)
    ax.plot(_xvals, _r2vals[2][0, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[2][1, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[2][2, :], lw=3, alpha=0.5)
    # ax.plot(_xvals, _r2vals[2][3, :], lw=3, alpha=0.5)
    ax.set_xlim(1, len(dur))
    ax.set_xticks(_xvals)
    ax.set_xticklabels(dur);
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Duration (sec)", fontsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.set_title("LDLJA", fontsize=14)
    # ax.legend(prop={'size': 12}, frameon=False)

    ax = fig.add_subplot(235)
    ax.plot(_xvals, _r2vals[3][0, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[3][1, :], lw=3, alpha=0.5)
    ax.plot(_xvals, _r2vals[3][2, :], lw=3, alpha=0.5)
    # ax.plot(_xvals, _r2vals[3][3, :], lw=3, alpha=0.5)
    ax.set_xlim(1, len(dur))
    ax.set_xticks(_xvals)
    ax.set_xticklabels(dur);
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Duration (sec)", fontsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.set_title("LDLJV-WOM", fontsize=14)
    # ax.legend(prop={'size': 12}, frameon=False, bbox_to_anchoYYeYesr=(1.1, 1.05))

    ax = fig.add_subplot(236)
    ax.plot(_xvals, _r2vals[4][0, :], lw=3, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[0]))
    ax.plot(_xvals, _r2vals[4][1, :], lw=3, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[1]))
    ax.plot(_xvals, _r2vals[4][2, :], lw=3, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[2]))
    # ax.plot(_xvals, _r2vals[4][3, :], lw=3, alpha=0.5, label="{0:0.0f} $\deg$".format(perr[3]))
    ax.set_xlim(1, len(dur))
    ax.set_xticks(_xvals)
    ax.set_xticklabels(dur);
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Duration (sec)", fontsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.set_title("LDLJA-WOM", fontsize=14)
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()

    return fig

def gen_correlation_versus_sgr(data):
    _sgrth = [1.0, 1.05, 1.1, 1.2, 1.5, 1.75, 2.0]
    _cols = [['sparc', 'sparcr'],
             ['ldljv', 'ldljrv'],
             ['ldlja', 'ldljra']]
    _cols1 = [['sparc', 'sparcr-wom'],
              ['ldljv', 'ldljrv-wom'],
              ['ldlja', 'ldljra-wom']]
    _r2vals = [np.zeros(len(_sgrth)),
               np.zeros(len(_sgrth)),
               np.zeros(len(_sgrth))]
    _r2vals1 = [np.zeros(len(_sgrth)),
                np.zeros(len(_sgrth)),
                np.zeros(len(_sgrth))]
    for i, _s in enumerate(_sgrth):
        _sgrinx = data['sgr'] > _s
        for k in range(len(_cols)):
            _r2vals[k][i] = stats.pearsonr(data[_cols[k][0]][_sgrinx],
                                           data[_cols[k][1]][_sgrinx])[0]
            _r2vals1[k][i] = stats.pearsonr(data[_cols1[k][0]][_sgrinx],
                                            data[_cols1[k][1]][_sgrinx])[0]
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(121)
    _xvals = np.arange(1, len(_sgrth)+1)
    ax.plot(_sgrth, _r2vals[0], lw=3, alpha=0.5, label="SPARC")
    ax.plot(_sgrth, _r2vals[1], lw=3, alpha=0.5, label="LDLJV")
    ax.plot(_sgrth, _r2vals[2], lw=3, alpha=0.5, label="LDLJA")
#     ax.set_xlim(1, len(_sgrth))
    ax.set_ylim(0.0, 1.05)
#     ax.set_xticks(_xvals[::2])
#     ax.set_xticklabels(_sgrth[::2]);
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.set_xlabel("SGR", fontsize=14)
    ax.set_title("Reconst.", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    
    ax = fig.add_subplot(122)
    _xvals = np.arange(1, len(_sgrth)+1)
    _xvals = np.arange(1, len(_sgrth)+1)
    ax.plot(_sgrth, _r2vals1[0], lw=3, alpha=0.5, label="SPARC")
    ax.plot(_sgrth, _r2vals1[1], lw=3, alpha=0.5, label="LDLJV")
    ax.plot(_sgrth, _r2vals1[2], lw=3, alpha=0.5, label="LDLJA")
#     ax.set_xlim(1, len(_sgrth))
#     ax.set_xticks(_xvals[::2])
    ax.set_ylim(0.0, 1.05)
#     ax.set_xticklabels(_sgrth[::2]);
    ax.set_xlabel("SGR", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title("Reconst. w/o mean", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.legend(prop={'size': 12}, frameon=False)

    plt.tight_layout()

    return fig


def gen_relerr_compare_plot(data, params, meas1, meas2, meas3, title):
  fig = plt.figure(figsize=(6, 3))

  # Results for LDLJ Velocity
  ax = fig.add_subplot(111)
  _ldljvcomp = get_relerr_for_diff_dur(data, params, meas1, meas2)
  _ldljvcomp_wom = get_relerr_for_diff_dur(data, params, meas1, meas3)
  _ldljvcompgroup = [[_l1, _l2] for _l1, _l2 in zip(_ldljvcomp, _ldljvcomp_wom)]

  # Combined data
  mpl.style.use("default")
  # Without mean subtraction
  # clr = sb.xkcd_rgb['pale red']
  clr = (0.6, 0.6, 0.6)
  _alpha = 0.5
  bp = plt.boxplot(_ldljvcomp, positions=[1, 4, 7, 10], widths=0.8, notch=True,
                   sym='', vert=1, whis=1.5, patch_artist=True,
                   boxprops=dict(linestyle='-', linewidth=0.5, color=clr),
                   medianprops=dict(linestyle='-', linewidth=1.0, color='2.0'),
                   whiskerprops=dict(color=clr, alpha=0.6, lw=1.25),
                   capprops=dict(color=clr, alpha=0.6, lw=1.25),
                   flierprops=dict(color=clr, alpha=0.6, lw=1.25, markeredgecolor=clr));
  for i in range(len(params.Dur)):
    bp['boxes'][i].set_facecolor(clr)
    bp['boxes'][i].set_alpha(_alpha)

  # With mean subtraction
  # clr = sb.xkcd_rgb['denim blue']
  clr = (0.2, 0.2, 0.2)
  _alpha = 0.5
  bp = plt.boxplot(_ldljvcomp_wom, positions=[2, 5, 8, 11], widths=0.8, notch=True,
                   sym='', vert=1, whis=1.5, patch_artist=True,
                   boxprops=dict(linestyle='-', linewidth=0.5, color=clr),
                   medianprops=dict(linestyle='-', linewidth=1.0, color='2.0'),
                   whiskerprops=dict(color=clr, alpha=0.6, lw=1.25),
                   capprops=dict(color=clr, alpha=0.6, lw=1.25),
                   flierprops=dict(color=clr, alpha=0.6, lw=1.25, markeredgecolor=clr));
  for i in range(len(params.Dur)):
    bp['boxes'][i].set_facecolor(clr)
    bp['boxes'][i].set_alpha(_alpha)

  ax.set_xticklabels(params.Dur, fontsize=12)
  ax.set_xticks([1.5, 4.5, 7.5, 10.5])
  ax.tick_params(axis='both', which='major', labelsize=12)

  ax.set_xlabel("Duration (s)", fontsize=12)
  ax.set_ylabel("Error (%)", fontsize=12)

  ax.grid(color='0.9', linestyle='-', linewidth=0.5)
  ax.set_title(title)

  return fig


def get_relerr_for_diff_dur(data, params, meas1, meas2):
  _data = [ [] ] * len(params.Dur)
  for _i, _d in enumerate(params.Dur):
    _durinx = data['dur'] == _d
    _data[_i] = list(100 * (data.loc[_durinx, meas2] - data.loc[_durinx, meas1]) / np.abs(data.loc[_durinx, meas1]))
  return _data


# Additional figures.
def compare_ldlja_terms(data):
    fig = plt.figure(figsize=(12, 3.5))

    # Duration factor
    ax = fig.add_subplot(131)
    lims = [-3.2, -0.8]
    ax.plot(lims, lims, '0.6', lw=1.0)
    ax.plot(data['ldlja-T'], data['ldljra-T'], 'o', alpha=0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Original", fontsize=14)
    ax.set_ylabel("Reconstruction", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.set_title("Duration term", fontsize=16)

    # Amplitude faactor
    ax = fig.add_subplot(132)
    lims = [-5, 15]
    ax.plot(lims, lims, '0.6', lw=1.0)
    ax.plot(data['ldlja-A'], data['ldljra-A'], '.', alpha=0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Original", fontsize=14)
    ax.set_ylabel("Reconstruction", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.set_title("Amplitude term", fontsize=16)

    # Jer factor
    ax = fig.add_subplot(133)
    lims = [-20, 4]
    ax.plot(lims, lims, '0.6', lw=1.0)
    ax.plot(data['ldlja-J'], data['ldljra-J'], '.', alpha=0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Original", fontsize=14)
    ax.set_ylabel("Reconstruction", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.set_title("Jerk term", fontsize=16)

    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=1.0)

    return fig


def compare_ldlja_terms_different_durations(data, params):
    fig = plt.figure(figsize=(12, 14))

    for i, _d in enumerate(params.Dur):
        _inx = data.dur == _d
        # Duration factor
        ax = fig.add_subplot(4, 3, 3 * i + 1)
        lims = [-7, -1]
        ax.plot(lims, lims, '0.6', lw=1.0)
        _inx2 = _inx2 = data.phierr == 50.0
        ax.plot(data[_inx & _inx2]['ldlja'], data[_inx & _inx2]['ldljra'], '.')
        _inx2 = _inx2 = data.phierr == 25.0
        ax.plot(data[_inx & _inx2]['ldlja'], data[_inx & _inx2]['ldljra'], '.')
        _inx2 = _inx2 = data.phierr == 5.0
        ax.plot(data[_inx & _inx2]['ldlja'], data[_inx & _inx2]['ldljra'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"LDLJ-A ({_d}s)", fontsize=16)

        # Amplitude faactor
        ax = fig.add_subplot(4, 3, 3 * i + 2)
        lims = [-7, 15]
        ax.plot(lims, lims, '0.6', lw=1.0)
        # ax.plot(data[_inx]['ldlja-A'], data[_inx]['ldljra-A'], '.', color='k')
        _inx2 = _inx2 = data.phierr == 50.0
        ax.plot(data[_inx & _inx2]['ldlja-A'] + data[_inx & _inx2]['ldlja-T'],
                data[_inx & _inx2]['ldljra-A'] + data[_inx & _inx2]['ldljra-T'], '.')
        _inx2 = _inx2 = data.phierr == 25.0
        ax.plot(data[_inx & _inx2]['ldlja-A'] + data[_inx & _inx2]['ldlja-T'],
                data[_inx & _inx2]['ldljra-A'] + data[_inx & _inx2]['ldljra-T'], '.')
        _inx2 = _inx2 = data.phierr == 5.0
        ax.plot(data[_inx & _inx2]['ldlja-A'] + data[_inx & _inx2]['ldlja-T'],
                data[_inx & _inx2]['ldljra-A'] + data[_inx & _inx2]['ldljra-T'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Amplitude + Duration ({_d}s)", fontsize=16)

        # Jer factor
        ax = fig.add_subplot(4, 3, 3 * i + 3)
        lims = [-20, 4]
        ax.plot(lims, lims, '0.6', lw=1.0)
        # ax.plot(data[_inx]['ldlja-J'], data[_inx]['ldljra-J'], '.', color='k')
        _inx2 = _inx2 = data.phierr == 50.0
        ax.plot(data[_inx & _inx2]['ldlja-J'], data[_inx & _inx2]['ldljra-J'], '.')
        _inx2 = _inx2 = data.phierr == 25.0
        ax.plot(data[_inx & _inx2]['ldlja-J'], data[_inx & _inx2]['ldljra-J'], '.')
        _inx2 = _inx2 = data.phierr == 5.0
        ax.plot(data[_inx & _inx2]['ldlja-J'], data[_inx & _inx2]['ldljra-J'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Jerk ({_d}s)", fontsize=16)

    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=1.0)

    return fig


def compare_ldlja_terms_different_durations_sgr(data, params, sgrth=1.05):
    fig = plt.figure(figsize=(12, 14))

    for i, _d in enumerate(params.Dur):
        _inx = data.dur == _d
        # Duration factor
        ax = fig.add_subplot(4, 3, 3 * i + 1)
        lims = [-7, -1]
        ax.plot(lims, lims, '0.6', lw=1.0)
        _inx2 = data.sgr < sgrth
        ax.plot(data[_inx & _inx2]['ldlja'], data[_inx & _inx2]['ldljra'], '.')
        _inx2 = data.sgr >= sgrth
        ax.plot(data[_inx & _inx2]['ldlja'], data[_inx & _inx2]['ldljra'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"LDLJ-A ({_d}s)", fontsize=16)

        # Amplitude faactor
        ax = fig.add_subplot(4, 3, 3 * i + 2)
        lims = [-7, 15]
        ax.plot(lims, lims, '0.6', lw=1.0)
        _inx2 = data.sgr < sgrth
        ax.plot(data[_inx & _inx2]['ldlja-A'] + data[_inx & _inx2]['ldlja-T'],
                data[_inx & _inx2]['ldljra-A'] + data[_inx & _inx2]['ldljra-T'], '.')
        _inx2 = data.sgr >= sgrth
        ax.plot(data[_inx & _inx2]['ldlja-A'] + data[_inx & _inx2]['ldlja-T'],
                data[_inx & _inx2]['ldljra-A'] + data[_inx & _inx2]['ldljra-T'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Amplitude + Duration ({_d}s)", fontsize=16)

        # Jer factor
        ax = fig.add_subplot(4, 3, 3 * i + 3)
        lims = [-20, 4]
        ax.plot(lims, lims, '0.6', lw=1.0)
        _inx2 = data.sgr < sgrth
        ax.plot(data[_inx & _inx2]['ldlja-J'], data[_inx & _inx2]['ldljra-J'], '.')
        _inx2 = data.sgr >= sgrth
        ax.plot(data[_inx & _inx2]['ldlja-J'], data[_inx & _inx2]['ldljra-J'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Jerk ({_d}s)", fontsize=16)

    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=1.0)

    return fig


def compare_ldlja_terms_different_durations2(data, params):
    fig = plt.figure(figsize=(12, 14))

    for i, _d in enumerate(params.Dur):
        _inx = data.dur == _d
        # Duration factor
        ax = fig.add_subplot(4, 3, 3 * i + 1)
        lims = [-7, -1]
        ax.plot(lims, lims, '0.6', lw=1.0)
        _inx = data.dur == _d
        _ix = np.where(_inx)[0]
        for _j, j in enumerate(_ix):
          _c = 0.2 + 0.6 * (_j / len(_ix))
          ax.plot(data.loc[j, 'ldlja'], data.loc[j, 'ldljra'], '.', color=f"{_c}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"LDLJ-A ({_d}s)", fontsize=16)

        # Amplitude faactor
        ax = fig.add_subplot(4, 3, 3 * i + 2)
        lims = [-7, 15]
        ax.plot(lims, lims, '0.6', lw=1.0)
        for _j, j in enumerate(_ix):
          _c = 0.2 + 0.6 * (_j / len(_ix))
          ax.plot(data.loc[j, 'ldlja-A'], data.loc[j, 'ldljra-A'], '.', color=f"{_c}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Amplitude + Duration ({_d}s)", fontsize=16)

        # Jer factor
        ax = fig.add_subplot(4, 3, 3 * i + 3)
        lims = [-20, 4]
        ax.plot(lims, lims, '0.6', lw=1.0)
        for _j, j in enumerate(_ix):
          _c = 0.2 + 0.6 * (_j / len(_ix))
          ax.plot(data.loc[j, 'ldlja-J'], data.loc[j, 'ldljra-J'], '.', color=f"{_c}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Jerk ({_d}s)", fontsize=16)

    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=1.0)

    return fig


def compare_ldljv_terms_dur_phierr(data, params):
    fig = plt.figure(figsize=(12, 14))

    for i, _d in enumerate(params.Dur):
        _inx = data.dur == _d
        # Duration factor
        ax = fig.add_subplot(4, 3, 3 * i + 1)
        lims = [-17, -5]
        ax.plot(lims, lims, '0.6', lw=1.0)
        _inx2 = _inx2 = data.phierr == 50.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro'], data[_inx & _inx2]['ldljvr-gyro'], '.')
        _inx2 = _inx2 = data.phierr == 25.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro'], data[_inx & _inx2]['ldljvr-gyro'], '.')
        _inx2 = _inx2 = data.phierr == 5.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro'], data[_inx & _inx2]['ldljvr-gyro'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"LDLJ-V ({_d}s)", fontsize=16)

        # Amplitude faactor
        ax = fig.add_subplot(4, 3, 3 * i + 2)
        lims = [0, 10]
        ax.plot(lims, lims, '0.6', lw=1.0)
        # ax.plot(data[_inx]['ldljv-gyro-A'], data[_inx]['ldljvr-gyro-A'], '.', color='k')
        _inx2 = _inx2 = data.phierr == 50.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro-A'], data[_inx & _inx2]['ldljvr-gyro-A'], '.')
        _inx2 = _inx2 = data.phierr == 25.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro-A'], data[_inx & _inx2]['ldljvr-gyro-A'], '.')
        _inx2 = _inx2 = data.phierr == 5.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro-A'], data[_inx & _inx2]['ldljvr-gyro-A'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Amplitude + Duration ({_d}s)", fontsize=16)

        # Jer factor
        ax = fig.add_subplot(4, 3, 3 * i + 3)
        lims = [-20, 4]
        ax.plot(lims, lims, '0.6', lw=1.0)
        # ax.plot(data[_inx]['ldljv-gyro-J'], data[_inx]['ldljvr-gyro-J'], '.', color='k')
        _inx2 = _inx2 = data.phierr == 50.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro-J'], data[_inx & _inx2]['ldljvr-gyro-J'], '.')
        _inx2 = _inx2 = data.phierr == 25.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro-J'], data[_inx & _inx2]['ldljvr-gyro-J'], '.')
        _inx2 = _inx2 = data.phierr == 5.0
        ax.plot(data[_inx & _inx2]['ldljv-gyro-J'], data[_inx & _inx2]['ldljvr-gyro-J'], '.')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Original", fontsize=14)
        ax.set_ylabel("Reconstruction", fontsize=14)
        ax.grid(color='0.7', linestyle='--', linewidth=0.5)
        ax.set_title(f"Jerk ({_d}s)", fontsize=16)

    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=1.0)

    return fig


def ldlja_smootherr_vs_reconerr(data, sgrth, col1="ldlja", col2="ldljra"):
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(111)
    _inxnot = data['sgr'] < sgrth
    _errsgrnot = (data[_inxnot][col2] - data[_inxnot][col1]) / np.abs(data[_inxnot][col1])
    ax.plot((180 / np.pi) * data[_inxnot]['rnorm'], _errsgrnot, '.', alpha=0.6, label=f"SGR $<$ {sgrth}")
    ax.set_xlabel("$RMS(\\Vert \\delta\\mathbf{R} - \\mathbf{I}\\Vert_2)$", fontsize=14)
    ax.set_ylabel("Rel. reconst. error ($\epsilon$)", fontsize=14)
    ax.set_title(f"Reconstruction error vs. Rel. Error", fontsize=14)
    ax.set_xlim([0, 70])
    ax.set_ylim([-1.0, 1.2])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    
    # ax = fig.add_subplot(122)
    _inx = data['sgr'] >= sgrth
    _errsgr = (data[_inx][col2] - data[_inx][col1]) / np.abs(data[_inx][col1])
    ax.plot((180 / np.pi) * data[_inx]['rnorm'], _errsgr, '.', label=f"SGR $\geq$ {sgrth}")
    ax.set_xlabel("$RMS(\\Vert \\delta\\mathbf{R} - \\mathbf{I}\\Vert_2)$", fontsize=14)
    ax.legend(loc=1, frameon=False, fontsize=14, ncol=1)

    return fig
