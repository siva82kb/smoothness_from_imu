"""Module containing the necessary variables and functions for analysing
from reconstructed IMU data for complex reaching movements.
"""

import numpy as np
import os
import json
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd
import seaborn as sb
import progressbar as pb
import itertools
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("../scripts/")
import myrobotics as myrob
import virtualimu as vimu
from smoothness import sparc
from smoothness import log_dimensionless_jerk
from smoothness import log_dimensionless_jerk_imu
from smoothness import log_dimensionless_jerk_factors
from smoothness import log_dimensionless_jerk_imu_factors
import jerkcorrection as jc
import smoothness as pysm


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
    dt = 0.01
    fs = 1 / dt
    # Movement amplitude (cm)
    Amp = [5.0, 10.0, 15.0]
    Namp = len(Amp)
    # Different movement duration.
    # Dur = [1.0, 5.0, 10.0, 30.0]
    Dur = [2.0]
    Ndur = len(Dur)

    # # Different levels of angular rotation. The following term defines the 
    # # norm of the angular rotation vector. (deg / sec)
    # gyNorm = [0] + [5.0, 10.0, 20.0, 90.0]
    # NgyNormRep = 5

    # Data related stuff.
    datadir = "../virtualimu_data/data"
    files = glob.glob("{0}/*.csv".format(datadir))
    # Get parameters
    nVia = map(int, np.arange(0, 11, 1))
    NViaRep = 10

    # Total number of reps.
    # Ntotal = len(nVia) * len(Dur) * len(gyNorm) * NViaRep * NgyNormRep
    Ntotal = len(nVia) * len(Dur) * Namp * NViaRep

    # gravity (cm / sec)
    grav = np.array([[0, 0, -98.1]]).T

    # out directory
    outdir = "../virtualimu_data/reconstructed"
    fnamestr = "_".join(("{0}/rdata_v{{0:02d}}", "d{{1:02d}}",
                         "gn{{2:02d}}", "vr{{3:02d}}", "pr{{4:02d}}.csv"))
    fnamestr = fnamestr.format(outdir)
    # Data header.
    headcols = ("time",
                "ax", "ay", "az",
                "vx", "vy", "vz",
                "jx", "jy", "jz",
                "gyx", "gyy", "gyz",
                "axs", "ays", "azs",
                "gyxs", "gyys", "gyzs",
                "jxs", "jys", "jzs",
                "daxs", "days", "dazs",
                "dgyxs", "dgyys", "dgyzs",
                "d2gyxs", "d2gyys", "d2gyzs",
                "vxs", "vys", "vzs",
                "axs-wom", "ays-wom", "azs-wom",
                "vxs-wom", "vys-wom", "vzs-wom",
                "spd", "spds", "spds-wom",
                "R11", "R12", "R13",
                "R21", "R22", "R23",
                "R31", "R32", "R33",)
    # combs = itertools.product(('accl', 'vel'), range(4), ('x', 'y', 'z'))
    # headcols += tuple(["{0}hp{1}{2}".format(i, j, k)  for i, j, k in combs])
    # headcols += tuple(["spdrhp{0}".format(i) for i in xrange(4)])
    header = ", ".join(headcols)

    @staticmethod
    def write_params_file():
        # Details of the simulated movements
        params = {'Amp': Params.Amp,
                  'gravity': Params.grav.tolist(),
                  'dur': Params.Dur,
                  # 'gynorm': Params.gyNorm,
                  'dt': Params.dt,
                  'datadir': Params.datadir,
                  'files': Params.files,
                  'nVia': Params.nVia,
                  'Nrep': Params.NViaRep,
                  # 'NgyNormRep': Params.NgyNormRep,
                  'Ntotal': Params.Ntotal
                 }
        # Write the parameter file
        # Write params file
        with open("{0}/params.json".format(Params.outdir), "w") as fh:
            json.dump(params, fh, indent=4)


def get_all_indices(params):
    return itertools.product(xrange(len(params.nVia)),
                             xrange(len(params.Dur)),
                             xrange(len(params.gyNorm)),
                             xrange(params.NViaRep),
                             xrange(params.NgyNormRep))


def get_all_movement_data(params):
    # File name format.
    fname_str = "{0}/{{0:02d}}via_{{1:04d}}.csv".format(params.datadir)
    inxs = get_all_indices(params)
    _fac = np.pi / 180.
    for inx in inxs:
        nv = params.nVia[inx[0]]
        dur = params.Dur[inx[1]]
        gynorm = params.gyNorm[inx[2]]

        f = fname_str.format(nv, inx[3])
        # Read file.
        if os.path.isfile(f):
            data = pd.read_csv(f)
            yield {"inx": inx, "nv": nv, "dur": dur,
                   "gynorm": _fac * gynorm,
                   "N": len(data), "data": data, "fname": f}
        else:
            continue


def generate_save_imu_data(params):
    # define progress bar
    widgets = ['[', pb.Timer(), '] ', pb.Bar(), ' (', pb.ETA(), ')']
    bar = pb.ProgressBar(widgets=widgets, maxval=params.Ntotal)

    # Get generator for all data.
    _velinx = ['vx', 'vy', 'vz']
    _accinx = ['ax', 'ay', 'az']
    _jerinx = ['jx', 'jy', 'jz']
    alldata = get_all_movement_data(params)
    cnt = 0
    for d in alldata:
        cnt += 1

        # Time
        t = np.arange(0, d['dur'], d['dur'] / d['N']).reshape((d['N'], 1))

        # Scale acceleration and velocity based on the amplitude and
        # duration of the movements.
        # Get linear velocity, acceleration and jerk data.
        _k = params.Amp * (1 / d['dur'])
        vel = _k * np.array(d['data'][_velinx])
        spd = np.linalg.norm(vel, axis=1).reshape((d['N'], 1))
        _k = params.Amp * (1 / np.power(d['dur'], 2.0))
        accl = _k * np.array(d['data'][_accinx])
        _k = params.Amp * (1 / np.power(d['dur'], 3.0))
        jerk = _k * np.array(d['data'][_jerinx])

        # 1. Angular velocity o the sensor.
        gy = jc.gen_angular_velocity(t)
        _gyrnorm = np.linalg.norm(gy) / np.sqrt(d['N'])
        gy = d["gynorm"] * gy / _gyrnorm

        # 2. Get rotation matrix from angular velocity.
        rotmats = jc.get_rotmats(gy, t)

        # 3. Sensor data and data in sensor coordinates
        gy = gy.T
        accls = np.array([np.matmul(_R.T, _ac + params.grav[:, 0])
                          for _R, _ac in zip(rotmats, accl)])
        gys = np.array([np.matmul(_R.T, _gy)
                        for _R, _gy in zip(rotmats, gy)])
        jerks = np.array([np.matmul(_R.T, _j)
                          for _R, _j in zip(rotmats, jerk)])
        # Derivative of accelerometer data.
        daccls = np.vstack((np.array([0, 0, 0]), np.diff(accls, axis=0)))
        daccls = daccls / (params.dt * d['dur'])
        # Derivatives of gyroscope data
        dgys = np.vstack((np.zeros((1, 3)),
                          np.diff(gys, axis=0)))
        dgys = dgys / (params.dt * d['dur'])
        # Seconds dserivatives of gyroscope data
        d2gys = np.vstack((np.zeros((2, 3)),
                           np.diff(np.diff(gys, axis=0), axis=0)))
        d2gys = d2gys / np.power((params.dt * d['dur']), 2)
        # Velocity from accelerometer data
        vs = np.cumsum(accls, axis=0) * (params.dt * d['dur'])
        spds = np.linalg.norm(vs, axis=1).reshape((d['N'], 1))

        # 4. Calculate variables without mean
        accls_wom = accls - np.mean(accls, axis=0)
        vels_wom = np.cumsum(accls_wom, axis=0) * params.dt * d['dur']
        spds_wom = np.linalg.norm(vels_wom, axis=1).reshape((d['N'], 1))

        # Write data to a csv file.
        _fname = params.fnamestr.format(*d['inx'])
        _R = np.array(rotmats)
        _data = np.hstack((t, accl, vel, jerk, gy,
                           accls, gys, jerks,
                           daccls, dgys, d2gys,
                           vs, accls_wom, vels_wom,
                           spd, spds, spds_wom,
                           _R.reshape(d['N'], 9)))
        np.savetxt(_fname, _data, delimiter=",", fmt="%10.10f",
                   header=params.header)
        bar.update(cnt)


def _smoothsparc(sp, sps, spswom, fs):
    _ss, _, _ = sparc(sp, fs)
    _sss, _, _ = sparc(sps, fs)
    _ssswom, _, _ = sparc(spswom, fs)
    return _ss, _sss, _ssswom


def _smoothldljv(v, vs, vswom, fs):
    _sl = log_dimensionless_jerk(v, fs=fs, data_type="vel")
    _slr = log_dimensionless_jerk(vs, fs=fs, data_type="vel")
    _slrwom = log_dimensionless_jerk(vswom, fs=fs, data_type="vel")
    return _sl, _slr, _slrwom


def _smoothldlja(ac, acs, acswom, fs):
    _sl = log_dimensionless_jerk(ac, fs=fs, data_type="accl")
    _sls = log_dimensionless_jerk(acs, fs=fs, data_type="accl")
    _slswom = log_dimensionless_jerk(acswom, fs=fs, data_type="accl")
    return _sl, _sls, _slswom


def _smoothldljimu(ac, acs, gyros, fs):
    _sl = log_dimensionless_jerk_imu(ac, None, fs)
    _sls = log_dimensionless_jerk_imu(acs, gyros, fs)
    return _sl, _sls


def _smoothldljgyro(gyo, gys, dgys, d2gys, fs):
    _sl = log_dimensionless_jerk(gyo, fs=fs, data_type="vel")
    # Cross product.
    _cp = np.array([np.cross(_dgy, _gy) for _dgy, _gy in zip(dgys, gys)])
    # Jerk magnitude
    _gyjerk = np.sum(np.power(np.linalg.norm(d2gys - _cp, axis=1),2)) * (1 / fs)
    _gymag = np.max(np.linalg.norm(gys, axis=1))
    _dur = np.shape(gys)[0] * (1 / fs)
    _sls = -np.log(np.power(_dur, 3) * _gyjerk / np.power(_gymag, 2))
    return _sl, _sls


def _smoothldljgyro(gys, dgys, d2gys, fs):
    # Cross product.
    _cp = np.array([np.cross(_dgy, _gy) for _dgy, _gy in zip(dgys, gys)])
    # Jerk magnitude
    _gyjerk = np.sum(np.power(np.linalg.norm(d2gys - _cp, axis=1),2)) * (1 / fs)
    _gymag = np.max(np.linalg.norm(gys, axis=1))
    _dur = np.shape(gys)[0] * (1 / fs)
    _sls = -np.log(np.power(_dur, 3) * _gyjerk / np.power(_gymag, 2))
    return _sls


def get_smooth_vals(params):
    cols = ["via", "dur", "gynorm", "viarep",
            "gynormrep", "wg", "sgr",
            "lao", "las", "laws",
            "laserr", "lawserr",
            "lgyo", "lgys", "lgyserr"]

    # Indices to be used for accessing data
    _accinx = ['ax', 'ay', 'az']
    _accsinx = ['axs', 'ays', 'azs']
    _gyinx = ['gyx', 'gyy', 'gyz']
    _gysinx = ['gyxs', 'gyys', 'gyzs']
    _dgysinx = ['dgyxs', 'dgyys', 'dgyzs']
    _d2gysinx = ['d2gyxs', 'd2gyys', 'd2gyzs']
    smoothvals = pd.DataFrame(columns=cols)
    indices = get_all_indices(params)
    cnt = 0
    # define progress bar
    widgets = ['[', pb.Timer(), '] ', pb.Bar(), ' (', pb.ETA(), ')']
    bar = pb.ProgressBar(widgets=widgets, maxval=params.Ntotal)
    for inx in indices:
        _f = Params.fnamestr.format(*inx)
        if os.path.isfile(_f) is True:
            # read reconstrucuted data
            data = pd.read_csv(filepath_or_buffer=_f, delimiter=',')
            data = data.rename(columns=lambda x: x.strip())
            _N = len(data)
        else:
            continue

        # Sensor to gravity ratio
        _g = np.sqrt(_N) * np.linalg.norm(params.grav)
        _s = np.linalg.norm(np.array(data[_accsinx]))
        _sgr = _s / _g
        
        # w_rms
        _wg = np.linalg.norm(data[_gyinx]) / np.sqrt(_N)

        # Sampling frequency
        fs = Params.fs / params.Dur[inx[1]]
        
        # LDLJ of global linear acceleration and angular rotation
        lao = pysm.log_dimensionless_jerk(np.array(data[_accinx]),
                                          fs, data_type='accl', scale='ms')
        lgyo = pysm.log_dimensionless_jerk(np.array(data[_gyinx]),
                                           fs, data_type='vel', scale='max')
        
        # LDLJ of from IMU data
        las = pysm.log_dimensionless_jerk_imu(np.array(data[_accsinx]), None,
                                              params.grav, fs)
        laws = pysm.log_dimensionless_jerk_imu(np.array(data[_accsinx]),
                                               np.array(data[_gysinx]),
                                               params.grav, fs)
        laserr = 100 * (las - lao) / np.abs(lao)
        lawserr = 100 * (laws - lao) / np.abs(lao)
        
        # LDLJ of gyroscope data
        lgys = _smoothldljgyro(np.array(data[_gysinx]),
                               np.array(data[_dgysinx]),
                               np.array(data[_d2gysinx]), fs)
        lgyserr = 100 * (lgys - lgyo) / np.abs(lgyo)
        
        # Append to dataframe
        _data = {"via": [params.nVia[inx[0]]],
                 "dur": params.Dur[inx[1]],
                 "gynorm": [params.gyNorm[inx[2]]],
                 "viarep": [inx[3]],
                 "gynormrep": [inx[4]],
                 "wg": [_wg],
                 "sgr": [_sgr],
                 "lao": [lao],
                 "las": [las],
                 "laws": [laws],
                 "laserr": [laserr],
                 "lawserr": [lawserr],
                 "lgyo": [lgyo],
                 "lgys": [lgys],
                 "lgyserr": [lgyserr]
                }
        smoothvals = smoothvals.append(pd.DataFrame.from_dict(_data),
                                       ignore_index=True)
        cnt += 1
        bar.update(cnt)
    return smoothvals


if __name__ == '__main__':
    # Write params file.
    Params.write_params_file()

    # Load params.
    params = Params()