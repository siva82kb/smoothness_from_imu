"""Module containing functions for analysing the estimation of LDLJ using 
gyroscope correction. This module also contains functions used in the notebook
to understand jerk correction issues - 
"1a_Understanding_Jerk_Correction_Problems.ipynb".

Author: Sivakumar Balasubramanian
Date: 21 Jul 2019
"""

import sys
import numpy as np
import myrobotics as myrob
from scipy.signal import savgol_filter as sgfilt 
from datetime import datetime
import pandas as pd
import multiprocessing
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("scripts")
import smoothness as pysm


# Initialize random seed.
# random seed
_seed = (int(datetime.now().strftime('%Y%m%d')) +
         int(datetime.now().strftime('%H%M%S')))
np.random.seed(_seed)

def vec2ssmat(vec):
    vec = vec.reshape((3, 1))
    return np.array([[         0, -vec[2, 0],  vec[1, 0]],
                     [ vec[2, 0],          0, -vec[0, 0]],
                     [-vec[1, 0],  vec[0, 0],          0]])


def ssmat2skey(ssmat):
    return np.array([[(ss[2, 1] - ss[1, 2]) / 2.0,
                      (ss[0, 2] - ss[2, 0]) / 2.0,
                      (ss[1, 0] - ss[0, 1]) / 2.0]]).T


def rotangvel(w, t):
    _wnorm = np.linalg.norm(w)
    _ss = vec2ssmat(w / _wnorm)
    _wt = _wnorm * t
    if _wnorm == 0:
        return np.eye(3)
    else:
        return (np.eye(3) + np.sin(_wt) * _ss +
                (1 - np.cos(_wt)) * np.matmul(_ss, _ss))


def gen_random_polysine(t, ws=None, amps=None, phi=None):
    # Frequencies
    ws = (2 * np.pi * np.array([0, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
          if (ws is None) else ws)
    _N = len(ws)
    # Phases
    phi = np.random.randn(_N) if (phi is None) else phi
    # Amplitudes
    amps = _gen_random_amp(_N) if (amps is None) else amps
    # Out data
    _data = {'data': np.sum(np.array([amps * np.sin(ws * _t + phi)
                                      for _t in t]), axis=1),
             'ws': ws,
             'N': _N,
             'phi': phi,
             'amps': amps}
    return _data


def _gen_random_amp(N):
    _amps = np.random.randn(N) * np.power(np.arange(1, N + 1)[::-1], 2)
    return _amps / np.sum(np.abs(_amps))


def gen_angular_velocity(t):
    # Spherical coordinates of angular velocity
    _rtemp = gen_random_polysine(t)['data']
    _r = np.power(_rtemp, 2) * 2 * np.pi
    _teta = gen_random_polysine(t)['data']
    _phi = 2 * gen_random_polysine(t)['data']
    return np.array([[__r * np.sin(__t) * np.cos(__p),
                      __r * np.sin(__t) * np.sin(__p),
                      __r * np.cos(__t)] 
                     for __r, __t, __p in zip(_r, _teta, _phi)]).T


def gen_angular_velocity_with_all_factors(t):
    """Same as gen_angular_velocity, but returns all values used in the
    calculation."""
    # Spherical coordinates of angular velocity
    _rtemp = gen_random_polysine(t)
    _tetatemp = gen_random_polysine(t)
    _phitemp = gen_random_polysine(t)
    _r = np.power(_rtemp['data'], 2) * 2 * np.pi
    _teta = _tetatemp['data']
    _phi = 2 * _phitemp['data']
    # out data
    _data = {'data': np.array([[__r * np.sin(__t) * np.cos(__p),
                                __r * np.sin(__t) * np.sin(__p),
                                __r * np.cos(__t)] 
                                for __r, __t, __p in zip(_r, _teta, _phi)]).T,
             'rtemp': _rtemp, 'ttemp': _tetatemp, 'ptemp': _phitemp,
             'r': _r, 'teta': _teta, 'phi': _phi}
    return _data


def gen_comp_zero_mean(t, wi, ampx, phi, T):
    _w1, _w2 = get_smooth_startstop_window(t, T)
    _a = gen_random_polysine(t, wi, ampx, phi)
    _a1 = _w1 * _a
    _a2 = _w2 * _a
    _a2 = (_a2 / np.mean(_a2))  * np.mean(_a1)
    return _a1 - _a2


def gen_acco_jerko(t, wi, T):
    Nw = len(wi)
    ampx = _gen_random_amp(Nw)
    ampy = _gen_random_amp(Nw)
    ampz = _gen_random_amp(Nw)
    phi = np.random.randn(Nw)
    # Acceleration
    _acc = np.array([gen_random_polysine(t, wi, ampx, phi)['data'],
                     gen_random_polysine(t, wi, ampy, phi)['data'],
                     gen_random_polysine(t, wi, ampz, phi)['data']])
    _acc = _acc - np.mean(_acc, axis=1).reshape(3, 1)
    # Jerk
    _jerk = np.array([gen_random_polysine(t, wi, wi * ampx, phi + np.pi / 2)['data'],
                      gen_random_polysine(t, wi, wi * ampy, phi + np.pi / 2)['data'],
                      gen_random_polysine(t, wi, wi * ampz, phi + np.pi / 2)['data']])
    return _acc, _jerk


def gen_acco_jerko_with_all_factors(t, wi, T):
    """Same as gen_acco_jerko, but returns all values used in the
    calculation."""
    Nw = len(wi)
    ampx = _gen_random_amp(Nw)
    ampy = _gen_random_amp(Nw)
    ampz = _gen_random_amp(Nw)
    phi = np.random.randn(Nw)
    # Acceleration
    _accx = gen_random_polysine(t, wi, ampx, phi)
    _accy = gen_random_polysine(t, wi, ampy, phi)
    _accz = gen_random_polysine(t, wi, ampz, phi)
    _acc = np.array([_accx['data'],
                     _accy['data'],
                     _accz['data']])
    _acc = _acc - np.mean(_acc, axis=1).reshape(3, 1)
    # Jerk
    _jerkx = gen_random_polysine(t, wi, wi * ampx, phi + np.pi / 2)
    _jerky = gen_random_polysine(t, wi, wi * ampy, phi + np.pi / 2)
    _jerkz = gen_random_polysine(t, wi, wi * ampz, phi + np.pi / 2)
    _jerk = np.array([_jerkx['data'],
                      _jerky['data'],
                      _jerkz['data']])
    # Out data
    _data = {'accl': _acc, 'jerk': _jerk,
             'accx': _accx, 'accy': _accy, 'accz': _accy,
             'jerkx': _jerkx, 'jerky': _jerky, 'jerkz': _jerky}
    return _data


def get_smooth_startstop_window(t, T):
    dt = t[1] - t[0]
    _p1 = 0.1
    _p2 = 3 * _p1 + (0.8 - 3 * _p1) * np.random.rand()
    _p3 = 1 - _p1
    _T1 = _p1 * T
    _w1 = np.cumsum(np.exp(-np.power(2.5 * (t - _T1) / _T1, 2))) * dt
    _T2 = _p2 * T
    _w2 = np.cumsum(np.exp(-np.power(2.5 * (t - _T2) / _T1, 2))) * dt
    _T3 = _p3 * T
    _w3 = np.cumsum(np.exp(-np.power(2.5 * (t - _T3) / _T1, 2))) * dt

    _win1 = _w1 - _w2
    _win1 = _win1 / np.max(_win1)
    _win2 = _w2 - _w3
    _win2 = _win2 / np.max(_win2)

    return _win1, _win2


def get_rotmats(wo, t):
    dt = t[1] - t[0]
    N = len(t)
    rotmats = [ 0 ] * N
    rotmats[0] = np.eye(3)
    for i in range(1, N):
        _w = wo[:, i-1]
        _R = rotangvel(_w, dt)
        rotmats[i] = np.matmul(_R, rotmats[i-1])
    return rotmats


def gen_analytical_data(params):
    t = params[0]
    T = params[1]
    g = params[2]
    wi = params[3]
    ag = 1.0 if len(params) == 4 else params[4]
    wg = 1.0 if len(params) == 5 else params[5]

    dt = t[1] - t[0]
    # Global angular velociy
    sys.stdout.write("\r -")
    sys.stdout.flush()
    wo = gen_angular_velocity(t)
    _wonorm = np.linalg.norm(wo) / np.sqrt(np.shape(wo)[1])
    wo = wg * wo / _wonorm

    # Global acceleration and jerk
    sys.stdout.write("\b/")
    sys.stdout.flush()
    acclo, jerko = gen_acco_jerko(t, wi, T)
    acclo = ag * acclo
    jerko = ag * jerko

    # Rotation matrices
    sys.stdout.write("\b|")
    sys.stdout.flush()
    rotmats = get_rotmats(wo, t)

    # Accelerometer data
    sys.stdout.write("\b\\")
    sys.stdout.flush()
    accls = np.array([np.matmul(_R.T, _ao + g.T[0])
                      for _ao, _R in zip(acclo.T, rotmats)]).T
    jerks = np.array([np.matmul(_R.T, _jo)
                      for _jo, _R in zip(jerko.T, rotmats)]).T

    # Gyroscope data
    sys.stdout.write("\b-")
    sys.stdout.flush()
    ws = np.array([np.matmul(_R.T, _wo)
                   for _wo, _R in zip(wo.T, rotmats)]).T

    # Analytical derivative of accelerometer
    _temp = np.array([np.cross(_ws, _as)
                      for _as, _ws in zip(accls.T, ws.T)]).T
    daccls = jerks - _temp
    
    return (acclo, jerko, wo, accls, jerks, ws, daccls, rotmats)


def rotmat_deriv(rotmats, dt):
    _drmat = [ 0 ] * len(rotmats)
    _drmat[0] = np.zeros((3, 3))
    _drmat[1:] = [(_R2 - _R1) / dt for _R1, _R2 in zip(rotmats[:-1], rotmats[1:])]
    return _drmat


def get_angvel_from_rotmats(rotmats, dt):
    # Derivative of rotation matrices
    drotmats = rotmat_deriv(rotmats, dt)
    # skew symmetric matrix.
    _ssangvelo = [np.matmul(_dR, _R.T) for _dR, _R in zip(drotmats, rotmats)]
    # Angular velocity from skew symmetric matrix.
    angvelo = np.array([[(_s[2, 1] - _s[1, 2]) / 2.0,
                         (_s[0, 2] - _s[2, 0]) / 2.0,
                         (_s[1, 0] - _s[0, 1]) / 2.0] for _s in _ssangvelo]).T
    return angvelo, drotmats, _ssangvelo


def get_numerical_data(truth, dt, dsratio=1):
    # Numerically estimate angular velocity from rotation matrices
    wo_est, _, _ = get_angvel_from_rotmats(truth['R'][::dsratio], dt * dsratio)
    ws_est = np.array([np.matmul(_R.T, _woest)
                       for _R, _woest in zip(truth['R'], wo_est.T)]).T
    ws_est[:, 0] = truth['ws'][:, 0]
    
    # Numerical accelerometer derivative.
    _daccls = np.diff(truth['accls'][:, ::dsratio], axis=1) / (dt * dsratio)
    daccls_est = np.hstack((np.zeros((3,1)), _daccls))
    daccls_est[:, 0] = truth['daccls'][:, 0]
    
    return (wo_est, ws_est, daccls_est)


def _ldljo(acclo, jerko, dt, T):
    # Global LDLJ
    _N = np.shape(acclo)[1]
    _aoms = np.power(np.linalg.norm(acclo), 2) / _N
    _jo2 = np.sum(np.power(np.linalg.norm(jerko, axis=0), 2)) * dt
    _lo = - np.log((T / _aoms) * _jo2)
    return {'T': - np.log(T),
            'amax': np.log(_aoms),
            'j2': - np.log(_jo2),
            'ldlj': _lo}


def _ldljs(accls, daccls, dt, T, g):
    _N = np.shape(accls)[1]
    _asms = np.power(np.linalg.norm(accls), 2) / _N
    _asms = _asms - np.linalg.norm(g)
    _js2 = np.sum(np.power(np.linalg.norm(daccls, axis=0), 2)) * dt
    _ls = - np.log((T / _asms) * _js2)
    return {'T': - np.log(T),
            'amax': np.log(_asms),
            'j2': - np.log(_js2),
            'ldlj': _ls}


def _ldljsc(accls, daccls, ws, dt, T, g):
    _N = np.shape(accls)[1]
    _asms = np.power(np.linalg.norm(accls), 2) / _N
    _asms = _asms - np.linalg.norm(g)
    # Get corrected jerk
    _jsc = _get_corrected_jerk(accls, daccls, ws)
    _jsc2 = np.sum(np.power(np.linalg.norm(_jsc, axis=0), 2)) * dt
    _lsc = -np.log((T / _asms) * _jsc2)
    return {'T': - np.log(T),
            'amax': np.log(_asms),
            'j2': - np.log(_jsc2),
            'ldlj': _lsc}


def _get_corrected_jerk(accls, daccls, ws):
    # Accl and Gyro cross product for jerk mag. correction
    _awcross = np.array([np.cross(_as, _ws) for _as, _ws in zip(accls.T, ws.T)]).T
    return daccls - _awcross


def _get_ldlj_smoothness(acclo, jerko, wo, accls, ws, dt, T, g):
    # Calculate LDLJ for the movement.
    # 1. Global LDLJ
    lo = _ldljo(acclo, jerko, dt, T)

    # 2. Sensor LDLJ
    daccls = np.hstack((np.zeros((3,1)), np.diff(accls, axis=1) / dt))
    ls = _ldljs(accls, daccls, dt, T, g)

    #2a. Sensr LDLJ without gyro
    ls1 = {'ldlj': sm.log_dimensionless_jerk_imu(accls.T, None, g,  1.0 / dt)}

    # 3. Corrected sensor LDLJ
    lsc = _ldljsc(accls, daccls, ws, dt, T, g)

    # 3a. Corrected sensor LDLJ using function.
    lsc1 = {'ldlj': sm.log_dimensionless_jerk_imu(accls.T, ws.T, g, 1.0 / dt)}
    
    return lo, ls, ls1, lsc, lsc1


def get_movement_ldlj_smoothness(params):
    t = params[0]
    T = params[1]
    g = params[2]
    wi = params[3]
    ag = 1.0 if len(params) == 4 else params[4]
    wg = 1.0 if len(params) == 5 else params[5]
    dsratio = 1 if len(params) == 6 else params[6]

    dt = t[1] - t[0]
    # Global angular velociy
    sys.stdout.write("\b-")
    sys.stdout.flush()
    wo = gen_angular_velocity(t)
    _wonorm = np.linalg.norm(wo) / np.sqrt(np.shape(wo)[1])
    wo = wg * wo / _wonorm

    # Global acceleration and jerk
    sys.stdout.write("\b/")
    sys.stdout.flush()
    acclo, jerko = gen_acco_jerko(t, wi, T)
    acclo = ag * acclo
    jerko = ag * jerko

    # Rotation matrices
    sys.stdout.write("\b|")
    sys.stdout.flush()
    rotmats = get_rotmats(wo, t)

    # Accelerometer data
    sys.stdout.write("\b\\")
    sys.stdout.flush()
    accls = np.array([np.matmul(_R.T, _ao + g.T[0])
                      for _ao, _R in zip(acclo.T, rotmats)]).T

    # Gyroscope data
    sys.stdout.write("\b-")
    sys.stdout.flush()
    ws = np.array([np.matmul(_R.T, _wo)
                   for _wo, _R in zip(wo.T, rotmats)]).T

    # Estimate smoothness
    daccls = np.hstack((np.zeros((3,1)), np.diff(accls, axis=1) / dt))
    return  {'smooth': _get_ldlj_smoothness(acclo, jerko, wo, accls, ws, dt, T, g),
             'data': (acclo, jerko, wo, accls, ws, daccls)}


def get_movement_ldlj_smoothness_all_factors(params):
    t = params[0]
    T = params[1]
    g = params[2]
    wi = params[3]
    ag = 1.0 if len(params) == 4 else params[4]
    wg = 1.0 if len(params) == 5 else params[5]

    dt = t[1] - t[0]
    # Global angular velociy
    # sys.stdout.write("\b-")
    sys.stdout.flush()
    wo_allfac = gen_angular_velocity_with_all_factors(t)
    wo = wo_allfac['data']
    _wonorm = np.linalg.norm(wo) / np.sqrt(np.shape(wo)[1])
    wo = wg * wo / _wonorm

    # Global acceleration and jerk
    # sys.stdout.write("\b/")
    sys.stdout.flush()
    accjerk_allfac = gen_acco_jerko_with_all_factors(t, wi, T)
    acclo, jerko = accjerk_allfac['accl'], accjerk_allfac['jerk']
    acclo = ag * acclo
    jerko = ag * jerko

    # Rotation matrices
    # sys.stdout.write("\b|")
    sys.stdout.flush()
    rotmats = get_rotmats(wo, t)

    # Accelerometer data
    # sys.stdout.write("\b\\")
    sys.stdout.flush()
    accls = np.array([np.matmul(_R.T, _ao + g.T[0])
                      for _ao, _R in zip(acclo.T, rotmats)]).T

    # Gyroscope data
    # sys.stdout.write("\b-")
    sys.stdout.flush()
    ws = np.array([np.matmul(_R.T, _wo)
                   for _wo, _R in zip(wo.T, rotmats)]).T

    # Estimate smoothness
    daccls = np.hstack((np.zeros((3,1)), np.diff(accls, axis=1) / dt))
    
    return  {'smooth': _get_ldlj_smoothness(accelerometer, jerko, wo, accls, ws, dt, T),
             'data': (acclo, jerko, wo, accls, ws, daccls),
             'angvel': wo_allfac, 'accjerk': accjerk_allfac}


def generate_jerkcorrection_summary(t, T, g, wi, Nmoves, ag, wg, dsratio, outdir, results):
    # Generate simulated data
    proc = multiprocessing.Pool(1)
    
    # Raw data header
    _header = ', '.join(('time',
                         'aox', 'aoy', 'aoz',
                         'jox', 'joy', 'joz',
                         'wox', 'woy', 'woz',
                         'asx', 'asy', 'asz',
                         'wsx', 'wsy', 'wsz',
                         'dasx', 'dasy', 'dasz'))

    # Run the simulation for 1000 different movements.
    sys.stdout.write("\r SGR: {0} | Wg: {1} | ".format(ag, wg))
    for i in range(Nmoves):
        sys.stdout.write("\r SGR: {0} | Wg: {1} | {2:03d}".format(ag, wg, i))
        _res = proc.map(get_movement_ldlj_smoothness, [[t, T, g, wi, ag, wg, dsratio]])[0]
        lo, ls, lsf, lsc, lscf = _res['smooth']
        aco, jo, wo, acs, ws, dacs = _res['data']
        # Update data.
        _N = np.shape(wo)[1]
        _wg = np.linalg.norm(wo) / np.sqrt(_N)
        _sgr = np.linalg.norm(acs) / (np.sqrt(_N) * np.linalg.norm(g))
        _df = pd.DataFrame.from_dict({'move': [i],
                                      'lo': [lo['ldlj']],
                                      'ls': [ls['ldlj']],
                                      'lsc': [lsc['ldlj']],
                                      'To': [lo['T']],
                                      'Ts': [ls['T']],
                                      'Tsc': [ls['T']],
                                      'amaxo': [lo['amax']],
                                      'amaxs': [ls['amax']],
                                      'amaxsc': [lsc['amax']],
                                      'j2o': [lo['j2']],
                                      'j2s': [ls['j2']],
                                      'j2sc': [lsc['j2']],
                                      'wgain': [wg], 'again': [ag],
                                      '_wgain': [wg], 
                                      'sgr': [_sgr]})
        results = results.append(_df, ignore_index=True, sort=True)
        # Save raw data
        fname = "{0}/rawdata/{1}_{2}_{3}.csv".format(outdir, ag, wg, i)
        np.savetxt(fname, np.hstack((t.reshape((len(t), 1)), aco.T, jo.T, wo.T,
                                     acs.T, ws.T, dacs.T)),
                   fmt='%10.8f', delimiter=', ', newline='\n', header=_header)
    return results


def get_original_movement(params):
    t = params[0]
    T = params[1]
    g = params[2]
    wi = params[3]
    ag = 1.0 if len(params) == 4 else params[4]
    wg = 1.0 if len(params) == 5 else params[5]

    dt = t[1] - t[0]
    
    # Global angular velociy
    wo = gen_angular_velocity(t)
    _wonorm = np.linalg.norm(wo) / np.sqrt(np.shape(wo)[1])
    wo = wg * wo / _wonorm

    # Global acceleration and jerk
    acclo, jerko = gen_acco_jerko(t, wi, T)
    acclo = ag * acclo
    jerko = ag * jerko
    
    return  acclo, jerko, wo


def get_sensor_data(t, acclo, wo, jerko, g, dsratio):
    _dt = (t[1] - t[0]) * dsratio    
    # Rotation matrices
    rotmats = get_rotmats(wo, t)[::dsratio]
    
    # Accelerometer data
    accls = np.array([np.matmul(_R.T, _ao + g.T[0])
                      for _ao, _R in zip(acclo[:, ::dsratio].T, rotmats)]).T
    # Gyroscope data
    ws = np.array([np.matmul(_R.T, _wo)
                   for _wo, _R in zip(wo[:, ::dsratio].T, rotmats)]).T

    return t[::dsratio], accls, ws


def gen_original_and_sensor_data(params):
    t = params[0]
    g = params[2]
    dsratio = params[6]
    
    # Original data
    acclo, jerko, wo = get_original_movement(params)
    
    # Sensor data
    ts, accls, ws = get_sensor_data(t, acclo, wo, jerko, g, dsratio)
    
    return acclo, jerko, wo, ts, accls, ws


def save_rawdata(ag, wg, i, logdetails, t, rawdata):
    _strorigfile = "{0}/rawdata/originaldata_{1}_{2}_{3}.csv"
    _strsensfile = "{0}/rawdata/sensordata_{1}_{2}_{3}.csv"
    # Save original data
    fname = _strorigfile.format(logdetails['outdir'], ag, wg, i)
    _data = np.hstack((t.reshape((len(t), 1)), rawdata[0].T,
                       rawdata[1].T, rawdata[2].T))
    np.savetxt(fname, _data, fmt='%10.8f', delimiter=', ',
               newline='\n', header=logdetails['orighead'])
    # Save sensor data
    fname = _strsensfile.format(logdetails['outdir'], ag, wg, i)
    _data = np.hstack((rawdata[3].reshape((len(rawdata[3]), 1)),
                       rawdata[4].T, rawdata[5].T))
    np.savetxt(fname, _data, fmt='%10.8f', delimiter=', ',
                   newline='\n', header=logdetails['senshead'])


# def generate_imuldlj_summary(t, T, g, wi, Nmoves, ag, wg, dsratio, logdetails, results):
#     # Generate simulated data
#     proc = multiprocessing.Pool(1)
    
#     # Run the simulation for 1000 different movements.
#     for i in range(Nmoves):
#         #  sys.stdout.write(" SGR: {0} | Wg: {1} | {2:03d}".format(ag, wg, i))
#         # params
#         params = [t, T, g, wi, ag, wg, dsratio]
#         # rawdata = proc.map(gen_original_and_sensor_data, [params])[0]
#         rawdata = gen_original_and_sensor_data(params)
        
#         # Save data.
#         # save_rawdata(ag, wg, i, logdetails, t, rawdata)
        
#         # True LDLJ smoothness
#         fs = 1 / (t[1] - t[0])
#         acclo, wo = rawdata[0].T, rawdata[2].T
#         loa = pysm.log_dimensionless_jerk(acclo, fs, data_type='accl', scale='ms')
#         loafac = pysm.log_dimensionless_jerk_factors(acclo, fs, data_type='accl', scale='ms')
                
#         ts, accls, ws = rawdata[3], rawdata[4].T, rawdata[5].T
#         fs = 1 / (ts[1] - ts[0])
        
#         # Sensor LDLJ smoothness from just accelerometer
#         lsa = pysm.log_dimensionless_jerk_imu(accls, None, g, fs)
#         lsafac = pysm.log_dimensionless_jerk_imu_factors(accls, None, g, fs)
        
#         # Sensor LDLJ smoothness from just accelerometer and gyroscope
#         lsaw = pysm.log_dimensionless_jerk_imu(accls, ws, g, fs)
#         lsawfac = pysm.log_dimensionless_jerk_imu_factors(accls, ws, g, fs)
        
#         # True gyroscope smoothness
        
#         # Sensor smoothness from gyroscope
        
#         # Update data.
#         _N = np.shape(wo)[0]
#         _wg = np.linalg.norm(wo) / np.sqrt(_N)
#         _N = np.shape(accls)[0]
#         _sgr = np.linalg.norm(accls) / (np.sqrt(_N) * np.linalg.norm(g))
#         _df = pd.DataFrame.from_dict({'move': [i],
#                                       'loa': [loa],
#                                       'lsa': [lsa],
#                                       'lsaw': [lsaw],
#                                       'To': [loafac[0]],
#                                       'Tsa': [lsafac[0]],
#                                       'Tsaw': [lsawfac[0]],
#                                       'aoms': [loafac[1]],
#                                       'asms': [lsafac[1]],
#                                       'aswms': [lsawfac[1]],
#                                       'jo': [loafac[2]],
#                                       'jsa': [lsafac[2]],
#                                       'jsaw': [lsawfac[2]],
#                                       'wg': [wg],
#                                       'ag': [ag],
#                                       '_wg': [_wg], 
#                                       'sgr': [_sgr]})
#         results = results.append(_df, ignore_index=True, sort=True)
#     return results


def generate_imuldlj_summary(t, T, g, wi, agvals, wgvals, dsratio, logdetails, results):
    # Generate simulated data
    proc = multiprocessing.Pool(1)
    
    # Run the simulation for 1000 different movements.
    for i, _vals in enumerate(np.array([agvals, wgvals]).T):
        ag, wg = _vals
        sys.stdout.write("\r{2}:: SGR: {0} | Wg: {1}".format(ag, wg, i))
        # params
        params = [t, T, g, wi, ag, wg, dsratio]
        # rawdata = proc.map(gen_original_and_sensor_data, [params])[0]
        rawdata = gen_original_and_sensor_data(params)
        
        # Save data.
        save_rawdata(ag, wg, i, logdetails, t, rawdata)
        
        # True LDLJ smoothness
        fs = 1 / (t[1] - t[0])
        acclo, wo = rawdata[0].T, rawdata[2].T
        loa = pysm.log_dimensionless_jerk(acclo, fs, data_type='accl', scale='ms')
        loafac = pysm.log_dimensionless_jerk_factors(acclo, fs, data_type='accl', scale='ms')
                
        ts, accls, ws = rawdata[3], rawdata[4].T, rawdata[5].T
        fs = 1 / (ts[1] - ts[0])
        
        # Sensor LDLJ smoothness from just accelerometer
        lsa = pysm.log_dimensionless_jerk_imu(accls, None, g, fs)
        lsafac = pysm.log_dimensionless_jerk_imu_factors(accls, None, g, fs)
        
        # Sensor LDLJ smoothness from just accelerometer and gyroscope
        lsaw = pysm.log_dimensionless_jerk_imu(accls, ws, g, fs)
        lsawfac = pysm.log_dimensionless_jerk_imu_factors(accls, ws, g, fs)
        
        # True gyroscope smoothness
        
        # Sensor smoothness from gyroscope
        
        # Update data.
        _N = np.shape(wo)[0]
        _wg = np.linalg.norm(wo) / np.sqrt(_N)
        _N = np.shape(accls)[0]
        _sgr = np.linalg.norm(accls) / (np.sqrt(_N) * np.linalg.norm(g))
        _df = pd.DataFrame.from_dict({'move': [i],
                                      'loa': [loa],
                                      'lsa': [lsa],
                                      'lsaw': [lsaw],
                                      'To': [loafac[0]],
                                      'Tsa': [lsafac[0]],
                                      'Tsaw': [lsawfac[0]],
                                      'aoms': [loafac[1]],
                                      'asms': [lsafac[1]],
                                      'aswms': [lsawfac[1]],
                                      'jo': [loafac[2]],
                                      'jsa': [lsafac[2]],
                                      'jsaw': [lsawfac[2]],
                                      'wg': [wg],
                                      'ag': [ag],
                                      '_wg': [_wg], 
                                      'sgr': [_sgr]})
        results = results.append(_df, ignore_index=True, sort=True)
    return results

def generate_ldlj_summary_plots(results, wgvals, sgth=1.1):
    lims = [-5.2, 2.2]

    fig = plt.figure(figsize=(12, 8))
    gs = mpl.gridspec.GridSpec(2, 3)

    _inx1 = (results.sgr <= sgth)
    titlestr = "SGR $\leq$ {0:0.1f}, $w_{{rms}}$ = {1:0.1f} rad/s"
    ax = plt.subplot(gs[0, 0])
    winx = 0
    plt.plot(lims, lims, 'k', alpha=0.3)
    _inx2 = (results.wgain == wgvals[winx])
    ax.plot(results.lo[_inx1 & _inx2], results.lsc[_inx1 & _inx2], 'o', alpha=0.2, label="LDLJ")
    ax.plot(results.lo[_inx1 & _inx2], results.ls[_inx1 & _inx2], 'o', alpha=0.2, label="LDLJ Corrected")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xticklabels([])
    ax.set_ylabel("LDLJ Sensor", fontsize=18)
    ax.set_title(titlestr.format(sgth, wgvals[winx]), fontsize=15)
    ax.legend(loc=2, prop={'size': 14})

    ax = plt.subplot(gs[0, 1])
    winx = 1
    plt.plot(lims, lims, 'k', alpha=0.3)
    _inx2 = (results.wgain == wgvals[winx])
    ax.plot(results.lo[_inx1 & _inx2], results.lsc[_inx1 & _inx2], 'o', alpha=0.2)
    ax.plot(results.lo[_inx1 & _inx2], results.ls[_inx1 & _inx2], 'o', alpha=0.2)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(titlestr.format(sgth, wgvals[winx]), fontsize=15)


    ax = plt.subplot(gs[0, 2])
    winx = 2
    plt.plot(lims, lims, 'k', alpha=0.3)
    _inx2 = (results.wgain == wgvals[winx])
    ax.plot(results.lo[_inx1 & _inx2], results.lsc[_inx1 & _inx2], 'o', alpha=0.2)
    ax.plot(results.lo[_inx1 & _inx2], results.ls[_inx1 & _inx2], 'o', alpha=0.2)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(titlestr.format(sgth, wgvals[winx]), fontsize=15)

    _inx1 = (results.sgr > sgth)
    titlestr = "SGR $>$ {0:0.1f}, $w_{{rms}}$ = {1:0.1f} rad/s"
    ax = plt.subplot(gs[1, 0])
    winx = 0
    plt.plot(lims, lims, 'k', alpha=0.3)
    _inx2 = (results.wgain == wgvals[winx])
    ax.plot(results.lo[_inx1 & _inx2], results.lsc[_inx1 & _inx2], 'o', alpha=0.2)
    ax.plot(results.lo[_inx1 & _inx2], results.ls[_inx1 & _inx2], 'o', alpha=0.2)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("LDLJ True", fontsize=18)
    ax.set_ylabel("LDLJ Sensor", fontsize=18)
    ax.set_title(titlestr.format(sgth, wgvals[winx]), fontsize=15)


    ax = plt.subplot(gs[1, 1])
    winx = 1
    plt.plot(lims, lims, 'k', alpha=0.3)
    _inx2 = (results.wgain == wgvals[winx])
    ax.plot(results.lo[_inx1 & _inx2], results.lsc[_inx1 & _inx2], 'o', alpha=0.2)
    ax.plot(results.lo[_inx1 & _inx2], results.ls[_inx1 & _inx2], 'o', alpha=0.2)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_yticklabels([])
    ax.set_xlabel("LDLJ True", fontsize=18)
    ax.set_title(titlestr.format(sgth, wgvals[winx]), fontsize=15)

    ax = plt.subplot(gs[1, 2])
    winx = 2
    plt.plot(lims, lims, 'k', alpha=0.3)
    _inx2 = (results.wgain == wgvals[winx])
    ax.plot(results.lo[_inx1 & _inx2], results.lsc[_inx1 & _inx2], 'o', alpha=0.2)
    ax.plot(results.lo[_inx1 & _inx2], results.ls[_inx1 & _inx2], 'o', alpha=0.2)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_yticklabels([])
    ax.set_xlabel("LDLJ True", fontsize=18)
    ax.set_title(titlestr.format(sgth, wgvals[winx]), fontsize=15);

    plt.tight_layout()
    
    return fig


# def generate_jerkcorrection_summary(t, T, g, wi, Nmoves, sgr, wg):
#     # Generate simulated data
#     proc = multiprocessing.Pool(1)

#     results = pd.DataFrame(columns=['move', 'lo', 'ls', 'lsc',
#                                     'j2o', 'j2s', 'j2sc',
#                                     'amaxo', 'amaxs', 'amaxsc'])
    
#     # Run the simulation for 1000 different movements.
#     sys.stdout.write("\r SGR: {0} | Wg: {1} | ".format(sgr, wg))
#     for i in range(Nmoves):
#         sys.stdout.write("\r SGR: {0} | Wg: {1} | {2:03d}".format(sgr, wg, i))
#         lo, ls, lsf, lsc, lscf = proc.map(get_movement_ldlj_smoothness, [[t, T, g, wi, sgr, wg]])[0]
#         _df = pd.DataFrame.from_dict({'move': [i],
#                                       'lo': [lo['ldlj']],
#                                       'ls': [ls['ldlj']],
#                                       'lsc': [lsc['ldlj']],
#                                       'j2o': [lo['j2']],
#                                       'j2s': [ls['j2']],
#                                       'j2sc': [lsc['j2']],
#                                       'amaxo': [lo['amax']],
#                                       'amaxs': [ls['amax']],
#                                       'amaxsc': [lsc['amax']]})
#         results = results.append(_df, ignore_index=True, sort=True)

#     # Save results.
#     _datadir = "../output/jerk_correction/"
#     results.to_csv("{0}/results_{1}_{2}.csv".format(_datadir, sgr, wg),
#                    columns=results.columns, index=False)
#     # Save figure.
#     lims = [-6, 0]
#     fig = figure(figsize=(7.5, 3))
#     gs = mpl.gridspec.GridSpec(1, 2)
#     ax = plt.subplot(gs[0, 0])
#     plot(lims, lims, 'k', lw=2, alpha=0.2)
#     plot(results.lo, results.ls, 'ro', alpha=0.1)
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     ax.set_xlabel("${}^{o}\lambda_{L}$", fontsize=16)
#     ax.set_ylabel("${}^{s}\lambda_{L}$", fontsize=16)
#     ax = plt.subplot(gs[0, 1])
#     plot(lims, lims, 'k', lw=2, alpha=0.2)
#     plot(results.lo, results.lsc, 'bo', alpha=0.1)
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     ax.set_xlabel("${}^{o}\lambda_{L}$", fontsize=16)
#     ax.set_ylabel("${}^{sc}\lambda_{L}$", fontsize=16)
#     fig.suptitle('SGR: {0}, Wgain: {1}'.format(sgr, wg), fontsize=18)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.85)
#     fig.savefig("{0}/ldljcomp_{1}_{2}.png".format(_datadir, sgr, wg), format="png", dpi=300)

def save_original_rawdata(t, aco, jo, wo, i, logdetails):
    # Save original data
    fname = logdetails['origbasename'].format(logdetails['outdir'], i)
    _data = np.hstack((t.reshape((len(t), 1)), aco.T, jo.T, wo.T))
    np.savetxt(fname, _data, fmt='%10.8f', delimiter=', ',
               newline='\n', header=logdetails['orighead'])    
    return fname


def gen_save_original_data(t, T, g, wi, agvals, wgvals,logdetails):
    # Generate simulated data
    proc = multiprocessing.Pool(1)

    # Run the simulation for 1000 different movements.
    filedetails = pd.DataFrame(columns=["N", 'wg', 'ag', 'fname'])
    for i, _vals in enumerate(np.array([agvals, wgvals]).T):
        ag, wg = _vals
        sys.stdout.write("\r{2}:: Ag: {0} | Wg: {1}".format(ag, wg, i))
        # params
        params = [t, T, g, wi, ag, wg]
        # Original data
        acclo, jerko, wo = get_original_movement(params)

        # Save data.
        _fn = save_original_rawdata(t, acclo, jerko, wo, i, logdetails)

        # Update file details
        filedetails = filedetails.append(pd.DataFrame.from_dict({"N": [i], "wg": [wg],
                                                                 "ag": [ag], "fname": [_fn]}),
                                         ignore_index=True, sort=True)
    filedetails.to_csv(logdetails['filedetails'], columns=filedetails.columns, index=False)


def gen_sensordata_ldljmsoothness(fdetails, logdetails, g, dsratio):
    results = pd.DataFrame(columns=['move', 'wg', 'ag', 'sgr',
                                    'arms', 'wrms',
                                    'loa', 'lsa', 'lsaw', 'To',
                                    'Tsa', 'Tsaw', 'aoms', 'asms',
                                    'aswms', 'jo', 'jsa', 'jsaw'])
    for _fr in fdetails.iterrows():
        sys.stdout.write("\r N: {0:04d} {1}".format(_fr[1]['N'], _fr[1]['fname']))
        # Read file data.
        origdata = pd.read_csv(_fr[1]["fname"], sep=", ", header=0, index_col=None).rename(columns={"# time": "time"})
        # Time and global kinematics
        t = np.array(origdata['time'])
        aco = np.array(origdata[['aox', 'aoy', 'aoz']]).T
        jo = np.array(origdata[['jox', 'joy', 'joz']]).T
        wo = np.array(origdata[['wox', 'woy', 'woz']]).T

        # Get sensor data
        ts, acs, ws = get_sensor_data(t, aco, wo, jo, g, dsratio)

        # True LDLJ smoothness
        fs = 1 / (t[1] - t[0])
        loa = pysm.log_dimensionless_jerk(aco.T[::dsratio, :], fs / dsratio,
                                          data_type='accl', scale='ms')
        loafac = pysm.log_dimensionless_jerk_factors(aco.T[::dsratio, :],
                                                     fs / dsratio,
                                                     data_type='accl',
                                                     scale='ms')

        # Sensor LDLJ smoothness from just accelerometer
        fss = 1 / (ts[1] - ts[0])
        lsa = pysm.log_dimensionless_jerk_imu(acs.T, None, g, fss)
        lsafac = pysm.log_dimensionless_jerk_imu_factors(acs.T, None, g, fss)

        # Sensor LDLJ smoothness from just accelerometer and gyroscope
        lsaw = pysm.log_dimensionless_jerk_imu(acs.T, ws.T, g, fss)
        lsawfac = pysm.log_dimensionless_jerk_imu_factors(acs.T, ws.T, g, fss)

        # Signal measures
        _N = np.shape(acs)[1]
        _sgr = np.linalg.norm(acs) / (np.sqrt(_N) * np.linalg.norm(g))
        _N = np.shape(aco)[1]
        _arms = np.linalg.norm(aco) / np.sqrt(_N)
        _N = np.shape(wo)[1]
        _wrms = np.linalg.norm(wo) / np.sqrt(_N)

        # Update data.
        _df = pd.DataFrame.from_dict({'move': [_fr[1]['N']],
                                      'wg': [_fr[1]['wg']],
                                      'ag': [_fr[1]['ag']],
                                      'sgr': [_sgr],
                                      'arms': [_arms],
                                      'wrms': [_wrms],
                                      'loa': [loa],
                                      'lsa': [lsa],
                                      'lsaw': [lsaw],
                                      'To': [loafac[0]],
                                      'Tsa': [lsafac[0]],
                                      'Tsaw': [lsawfac[0]],
                                      'aoms': [loafac[1]],
                                      'asms': [lsafac[1]],
                                      'aswms': [lsawfac[1]],
                                      'jo': [loafac[2]],
                                      'jsa': [lsafac[2]],
                                      'jsaw': [lsawfac[2]]})
        results = results.append(_df, ignore_index=True, sort=False)
    return results


def get_imu_data(wnorm, t, aco, wo):
    # Rotation matrices
    rotmats = [myrob.roty(wnorm * _t) for _t in t]

    # Accelerometer and gyroscope data
    acs = np.array([np.matmul(_R.T, _aco.reshape((3, 1))).T[0]
                    for _R, _aco in zip(rotmats, aco.T)]).T
    ws = np.array([np.matmul(_R.T, _wo.reshape((3, 1))).T[0]
                   for _R, _wo in zip(rotmats, wo.T)]).T
    return acs, ws


def get_sim_data_fixed_accl(dt, wnorm, ac=np.zeros((3, 1)), go=np.array([[0, 0, -9.81]]).T):
    sys.stdout.write("\rdT: {0}, w_norm:{1}".format(dt, wnorm))
    
    # Parameters
    T = 1
    t = np.arange(0, T, dt)
    N = len(t)
    
    # Zero acceleration
    # acl = np.matmul(np.diag(ac), np.ones((3, N)))
    acl = np.repeat(ac, N, axis=1)
    # Total acceleration in the global frame
    aco = acl + go
    # Analytical jerk in global frame
    jo = np.zeros((3, N))
    
    # Angular rotation in global frame
    wo = np.array([[0, wnorm, 0]]).T * np.ones((3, N))
    # Sensor parameters and angular rotation
    acs, ws = get_imu_data(wnorm, t, aco, wo)
    
    # Derivative of the accelerometer data
    dacs = np.diff(acs, axis=1) / dt
    dacs = np.hstack((dacs[:, 0].reshape((3, 1)), dacs))
    # Accl x gyro cross product
    awscp = np.array([np.cross(_acs, _ws) for _acs, _ws in zip(acs.T, ws.T)]).T
    
    return {'t': t, 'N': N, 'dt': dt, 'wnorm': wnorm,
            'acl': acl, 'go': go,
            'aco': aco, 'jo': jo, 'wo': wo,
            'acs': acs, 'ws': ws,
            'dacs': dacs, 'awscp': awscp
            }


def get_sim_data_fixed_jerk(dt, wnorm, ac=np.zeros((3, 1)), go=np.array([[0, 0, -9.81]]).T, jnorm=1):
    sys.stdout.write("\rdT: {0}, w_norm:{1}".format(dt, wnorm))
    
    # Parameters
    T = 1
    t = np.arange(0, T, dt)
    N = len(t)
    
    # Zero acceleration
    acl = jnorm * np.multiply(t, np.repeat(ac, N, axis=1))
    # Total acceleration in the global frame
    aco = acl + go
    # Analytical jerk in global frame
    jo = jnorm * np.repeat(ac, N, axis=1)
    
    # Angular rotation in global frame
    wo = np.array([[0, wnorm, 0]]).T * np.ones((3, N))
    # Sensor parameters and angular rotation
    acs, ws = get_imu_data(wnorm, t, aco, wo)
    
    # Derivative of the accelerometer data
    dacs = np.diff(acs, axis=1) / dt
    dacs = np.hstack((dacs[:, 0].reshape((3, 1)), dacs))
    # Accl x gyro cross product
    awscp = np.array([np.cross(_acs, _ws) for _acs, _ws in zip(acs.T, ws.T)]).T
    
    return {'t': t, 'N': N, 'dt': dt, 'wnorm': wnorm,
            'acl': acl, 'go': go,
            'aco': aco, 'jo': jo, 'wo': wo,
            'acs': acs, 'ws': ws,
            'dacs': dacs, 'awscp': awscp
            }


def get_sim_data_unicircmotion(dt, wnorm, anorm=1.0, go=np.array([[0, 0, -9.81]]).T):
    sys.stdout.write(f"\r w_norm:{wnorm}, a_norm:{anorm}, dT: {dt}")
    
    # Parameters
    T = 1
    # go = np.array([[0, 0, -9.81]]).T
    t = np.arange(0, T, dt)
    N = len(t)
    
    # Angular rotation in global frame
    wo = np.array([[0, wnorm, 0]]).T * np.ones((3, N))
    
    # Zero acceleration
    acl = anorm * np.array([-np.cos(wnorm * t),
                            np.zeros(N),
                            np.sin(wnorm * t)])
    # Total acceleration in the global frame
    aco = acl + go
    # Analytical jerk in global frame
    jo = anorm * np.array([wnorm * np.sin(wnorm * t),
                           np.zeros(N),
                           wnorm * np.cos(wnorm * t)])
    
    # Sensor parameters and angular rotation
    acs, ws = get_imu_data(wnorm, t, aco, wo)
    
    # Derivative of the accelerometer data
    dacs = np.diff(acs, axis=1) / dt
    dacs = np.hstack((dacs[:, 0].reshape((3, 1)), dacs))
    # Accl x gyro cross product
    awscp = np.array([np.cross(_acs, _ws) for _acs, _ws in zip(acs.T, ws.T)]).T
    
    # Signal to gravity ratio
    sgr = np.linalg.norm(acs) / (np.linalg.norm(go) * np.sqrt(N))
    
    return {'t': t, 'N': N, 'dt': dt,
            'anorm': anorm, 'wnorm': wnorm,
            'acl': acl, 'go': go,
            'aco': aco, 'jo': jo, 'wo': wo,
            'acs': acs, 'ws': ws,
            'dacs': dacs, 'awscp': awscp,
            'sgr': sgr
            }


def get_angvel_global(rotmats, dt):
    N = len(rotmats)
    drotmats = [(rotmats[i] - rotmats[i-1]) / dt for i in range(1, N)]
    drotmats = [drotmats[0]] + drotmats
    wssmat = [np.matmul(_dR, _R.T) for _dR, _R in zip(drotmats, rotmats)]
    return np.array([[0.5 * (_ws[1, 2] - _ws[2, 1]),
                      0.5 * (_ws[0, 2] - _ws[2, 0]),
                      0.5 * (_ws[1, 0] - _ws[0, 1])] for _ws in wssmat]).T


def get_sim_data_num(dt, wnorm, ac=np.zeros(3), go=np.array([[0, 0, -9.81]]).T):
    sys.stdout.write("\rdT: {0}, w_norm:{1}".format(dt, wnorm))
    
    # Parameters
    T = 1
    # go = np.array([[0, 0, -9.81]]).T
    t = np.arange(0, T, dt)
    N = len(t)
    
    # Zero acceleration
    acl = np.matmul(np.diag(ac), np.ones((3, N)))
    # Total acceleration in the global frame
    aco = acl + go
    # Analytical jerk in global frame
    jo = np.zeros((3, N))
    
    # Rotation matrices
    rotmats = [myrob.roty(wnorm * _t) for _t in t]

    # Estimate angular velocity in global frame.
    wo = get_angvel_global(rotmats, dt)

    # Accelerometer and gyroscope data
    acs = np.array([np.matmul(_R.T, _aco.reshape((3, 1))).T[0]
                    for _R, _aco in zip(rotmats, aco.T)]).T
    ws = np.array([np.matmul(_R.T, _wo.reshape((3, 1))).T[0]
                   for _R, _wo in zip(rotmats, wo.T)]).T

    # Derivative of the accelerometer data
    dacs = np.diff(acs, axis=1) / dt
    dacs = np.hstack((dacs[:, 0].reshape((3, 1)), dacs))
    awscp = np.array([np.cross(_acs, _ws) for _acs, _ws in zip(acs.T, ws.T)]).T

    return {'t': t, 'N': N, 'dt': dt, 'wnorm': wnorm,
            'acl': acl, 'go': go,
            'aco': aco, 'jo': jo, 'wo': wo,
            'acs': acs, 'ws': ws,
            'dacs': dacs, 'awscp': awscp
            }


def compare_jc_error_fixed_accl(simdata, dts, wnorms, wnormsd, anorm):
    # Plot summary
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(111)
    M, N = len(dts), len(wnorms)
    ylims = [-1, 20]
    cols = ['0.0', '0.2', '0.4', '0.6', '0.8']
    for i, s in enumerate(simdata):
        # Check if jerk magnitude is zero.
        if np.linalg.norm(s[0]['jo']) == 0:
            _err = np.array([np.linalg.norm(_s['dacs'] - _s['awscp']) / np.sqrt(_s['N']) for _s in s])
        else:
            _err = np.array([np.linalg.norm(_s['dacs'] - _s['awscp']) / np.sqrt(_s['N']) for _s in s])
            # _ylbl = "Jerk Error (%)"
        _ylbl = "Jerk Error ($m \\cdot s^{-3}$)"
        ax.plot(_err, label="{0:0.3f}s".format(dts[i]), lw=3, alpha=0.6, color=cols[i])
    ax.legend(loc=2, prop={'size': 12})
    # ax.set_ylim(ylims)
    ax.set_xticklabels(np.hstack(([0], wnormsd)))
    ax.set_ylabel(_ylbl, fontsize=14)
    ax.set_xlabel("Ang. Velocity ($deg \\cdot s^{-1}$)", fontsize=14)
    ax.set_title("Error in jerk correction $a_n = {0:0.02f}$".format(anorm), fontsize=14)
    plt.tight_layout()
    return fig


def compare_jerk_error_components(simdata, dts, wnorms, wnormsd, anorm, adir):
    fig = plt.figure(figsize=(15, 7.5))
    M, N = len(dts), len(wnorms)
    for i, s in enumerate(simdata):
        _err = np.array([(_s['dacs'] - _s['awscp']).T for _s in s])
        
        # x component
        ax = fig.add_subplot(3, M, i + 1)
        ax.boxplot(_err.T[0])
        ax.set_xticks([])
        if (i == 0):
            ax.set_ylabel("Error X ($m \\cdot s^{-3}$)")
        ax.set_title(f"$\\Delta t$ = {dts[i]}s")
        
        # y component
        ax = fig.add_subplot(3, M, M + i + 1)
        ax.boxplot(_err.T[1])
        ax.set_xticks([])
        if (i == 0):
            ax.set_ylabel("Error Y ($m \\cdot s^{-3}$)")
            
        # z component
        ax = fig.add_subplot(3, M, 2* M + i + 1)
        ax.boxplot(_err.T[2])
        ax.set_xlabel("Ang. Velocity (deg/sec)")
        ax.set_xticklabels(wnormsd)
        if (i == 0):
            ax.set_ylabel("Error Z ($m \\cdot s^{-3}$)")

    plt.suptitle(f"Acceleration ($a_n$) = {anorm} [{adir[0]}, {adir[1]}, {adir[2]}]")
    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.925, top=0.9,
                        wspace=0.3, hspace=0.2)
    return fig


def compare_jc_error_uni_circ(simdata_ucm, dts, wnorms, wnormsd, anorms):
    fig = plt.figure(figsize=(len(anorms) * 4, 3.5))
    for j in range(len(anorms)):
        ax = fig.add_subplot(1, len(anorms), j + 1)
        M, N = len(dts), len(wnorms)
        cols = ['0.0', '0.2', '0.4', '0.6', '0.8']
        for i, s in enumerate(simdata_ucm[j]):
            # Check if jerk magnitude is zero.
            if np.linalg.norm(s[0]['jo']) == 0:
                _err = __get_jc_errors_abs(s)
                _ylbl = "Jerk Error ($m \\cdot s^{-3}$)"
            else:
                _err = __get_jc_errors_prcnt(s)
                _ylbl = "Jerk Error (%)"
            # Sensor to gravity ratio
            _sgr = np.array([_s['sgr'] for _s in s])
            ax.plot(_err, label="{0:0.3f}s".format(dts[i]), lw=3, alpha=0.6, color=cols[i])
        if j == 0:
            ax.set_ylabel(_ylbl, fontsize=14)
            ax.legend(loc=2, prop={'size': 12}, frameon=False)
        ax.set_xlabel("Ang. Velocity ($deg \\cdot s^{-1}$)", fontsize=14)
        ax.set_title("JC error $a_n = {0:0.1f}$ (SGR: {1:0.1f})".format(anorms[j], _sgr[0]), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=13)
    plt.suptitle("Jerk Reconstruction Error in uniform circular motion", fontsize=16)
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.18, top = 0.8, wspace = 0.25, hspace = 0.25)

    return fig


def __get_jc_errors_abs(sdata):
    _err = []
    for _s in sdata:
        _jo = np.linalg.norm(_s['jo'], axis=0)
        _jest = np.linalg.norm(_s['dacs'] - _s['awscp'], axis=0)
        _Nsqrt = np.sqrt(_s['N'])
        _err.append(np.linalg.norm(_jest - _jo) / _Nsqrt)
    return np.array(_err)


def __get_jc_errors_prcnt(sdata):
    _err = []
    for _s in sdata:
        _jo = np.linalg.norm(_s['jo'], axis=0)
        _jest = np.linalg.norm(_s['dacs'] - _s['awscp'], axis=0)
        _Nsqrt = np.sqrt(_s['N'])
        _err.append(100 * np.linalg.norm((_jest - _jo) / _jo) / _Nsqrt)
    return np.array(_err)


def __plot_simdata_for_accl_norm_dir(ax, simdata, dts, wnorms, wnormsd, anorm,
                                    adir, xlbl=True, ylbl=True, dtype="abs"):
    # M, N = len(dts), len(wnorms)
    cols = ['0.0', '0.2', '0.4', '0.6', '0.8']
    for i, s in enumerate(simdata):
        # Check if jerk magnitude is zero.
        if dtype == "abs":
            # _err = np.array([np.linalg.norm(_s['dacs'] - _s['awscp']) / np.sqrt(_s['N']) for _s in s])
            # Get the errors for different angular rates
            _err = __get_jc_errors_abs(s)
            _ylbl = "Jerk Error ($m \\cdot s^{-3}$)"
        else:
            # _err = 100 * np.array([(np.linalg.norm(_s['dacs'] - _s['awscp']) - np.linalg.norm(_s['jo'])) / np.linalg.norm(_s['jo'])
            #                         for _s in s])
            _err = __get_jc_errors_prcnt(s)
            _ylbl = "Jerk Error (%)"
        ax.plot(_err, label="{0:0.3f}s".format(dts[i]), lw=3, alpha=0.6, color=cols[i])
    ax.legend(loc=2, prop={'size': 12})
    ax.set_xticklabels(np.hstack(([0], wnormsd)))
    if ylbl:
        ax.set_ylabel(_ylbl, fontsize=14)
    if xlbl:
        ax.set_xlabel("Ang. Velocity ($deg \\cdot s^{-1}$)", fontsize=14)
    ax.set_title(f"JC Err $a_n = {anorm} [{adir[0]}, {adir[1]}, {adir[2]}]$", fontsize=14)


def compare_jc_error_const_accl_diff_dir(simdata, dts, wnorms, wnormsd, anorms, accll, inx, dtype="abs"):
    # Plot summary
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(2, 3, 1)
    __plot_simdata_for_accl_norm_dir(ax, simdata[inx][0], dts, wnorms,
                                     wnormsd, anorms[inx], accll[0],
                                     xlbl=False, ylbl=True, dtype=dtype)
    ax = fig.add_subplot(2, 3, 2)
    __plot_simdata_for_accl_norm_dir(ax, simdata[inx][2], dts, wnorms,
                                     wnormsd, anorms[inx], accll[2],
                                     xlbl=False, ylbl=False, dtype=dtype)
    ax = fig.add_subplot(2, 3, 3)
    __plot_simdata_for_accl_norm_dir(ax, simdata[inx][4], dts, wnorms,
                                     wnormsd, anorms[inx], accll[4],
                                     xlbl=False, ylbl=False, dtype=dtype)
    ax = fig.add_subplot(2, 3, 4)
    __plot_simdata_for_accl_norm_dir(ax, simdata[inx][1], dts, wnorms,
                                     wnormsd, anorms[inx], accll[1],
                                     xlbl=True, ylbl=True, dtype=dtype)
    ax = fig.add_subplot(2, 3, 5)
    __plot_simdata_for_accl_norm_dir(ax, simdata[inx][3], dts, wnorms,
                                     wnormsd, anorms[inx], accll[3],
                                     xlbl=True, ylbl=False, dtype=dtype)
    ax = fig.add_subplot(2, 3, 6)
    __plot_simdata_for_accl_norm_dir(ax, simdata[inx][5], dts, wnorms,
                                     wnormsd, anorms[inx], accll[5],
                                     xlbl=True, ylbl=False, dtype=dtype)
    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.925, top=0.9, wspace=0.2, hspace=0.25)
    return fig
