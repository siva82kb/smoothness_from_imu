"""
Module for analysing smoothness measures on different data types, and to
generate summary plots.

Author: Sivakumar Balasubramanian
Date: 24 Oct 2019
"""

import sys
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

sys.path.append("../../scripts")
from movements import mjt_discrete_movement
from movements import gaussian_discrete_movement
from movements import generate_movement
from smoothness import sparc
from smoothness import log_dimensionless_jerk as LDLJ
from smoothness import log_dimensionless_jerk_factors as LDLJ_factors


# Matplotlib setting for plots.
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans'


def generate_simulated_movements(Ns, dT, Ts, ts, move_type):
    """Generates a set of movements with different submovement numbers
    and intervals."""
    moves = {'vel': [], 'accl': [], 'jerk': []}
    for k in moves.keys():
        for ni, n in enumerate(Ns):
            _temp = []
            for dt in dT:
                sys.stdout.write('\rNs: {0}, dT: {1}'.format(n, dt))
                t, m, _ = generate_movement(n, [1. / n] * n,
                                            [dt] * (n - 1), [Ts] * n,
                                            ts=ts, move_type=move_type,
                                            data_type=k)
                _temp.append(m)
            moves[k].append(_temp)
    return moves


def analyse_ldlj_from_different_signals(moves, Ns, dT, ts, amp_norm='max'):
    """Analyses LDLJ from different types of signals.
    """
    smooth_vals = {'vel': [], 'accl': []}
    scale_factors = {'vel': {'A': [], 'T': [], 'J': []},
                     'accl': {'A': [], 'T': [], 'J': []}}
    m_types = smooth_vals.keys()

    _str = '\rType: {0}, Ns: {1}, dT: {2}'
    # _s = {}
    for _type in m_types:
        _temp = np.zeros((len(Ns), len(dT)))
        _tempA = np.zeros((len(Ns), len(dT)))
        _tempT = np.zeros((len(Ns), len(dT)))
        _tempJ = np.zeros((len(Ns), len(dT)))
        for i in range(len(Ns)):
            _tmp = []
            _tmpA = []
            _tmpT = []
            _tmpJ = []
            for j in range(len(dT)):
                sys.stdout.write(_str.format(_type, Ns[i], dT[j]))
                _tmp.append(LDLJ(np.array([moves[_type][i][j]]).T,
                                 1/ts, data_type=_type, scale=amp_norm))
                _f = LDLJ_factors(np.array([moves[_type][i][j]]).T,
                                  1/ts, data_type=_type, scale=amp_norm)
                _tmpT.append(_f[0])
                _tmpA.append(_f[1])
                _tmpJ.append(_f[2])
            _temp[i, :] = _tmp
            _tempT[i, :] = _tmpT
            _tempA[i, :] = _tmpA
            _tempJ[i, :] = _tmpJ
        # Add to appropriate data type
        smooth_vals[_type] = _temp
        scale_factors[_type]['T'] = _tempT
        scale_factors[_type]['A'] = _tempA
        scale_factors[_type]['J'] = _tempJ
    return smooth_vals, scale_factors


def read_movement_data(data_dir, details):
    """Reads each data file and yields data and its details
    one by one."""
    Nvias = details.keys()
    for Nvia in Nvias:
        moves = details[Nvia]
        for Nmove, move in enumerate(moves):
            _data = pd.read_csv("{0}/{1}".format(data_dir, move['file']),
                                index_col=None)
            yield Nvia, Nmove, move, _data


def generate_summary_plot_for_movement(pdf, mdata, minfo, Nvia, Nmove, fs,
                                       v, v1, v2, a, a1, a2, j, j1, j2,
                                       sv, sa, sj, lv, la, lj):
    # plot data
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(231)
    _t = np.arange(0, len(v) * 1 / fs, 1 / fs)
    plt.plot(_t, mdata[['vx', 'vy', 'vz']][:-1], lw=0.5)
    plt.plot(_t, v, 'k', lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_title('Velocity')

    ax = fig.add_subplot(232)
    _t = np.arange(0, len(a) * 1 / fs, 1 / fs)
    plt.plot(_t, mdata[['ax', 'ay', 'az']][:-2], lw=0.5)
    plt.plot(_t, a, 'k', lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_title('Acceleration')

    ax = fig.add_subplot(233)
    _t = np.arange(0, len(j) * 1 / fs, 1 / fs)
    plt.plot(_t, mdata[['jx', 'jy', 'jz']][:-3], lw=0.5)
    plt.plot(_t, j, 'k', lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_title('Jerk')

    ax = fig.add_subplot(234)
    plt.plot(v1[0], v1[1])
    plt.plot(v2[0], v2[1])
    ax.set_xlim(0, 50.)
    ax.set_xlabel('Frequency (hz)')
    ax.set_title('SPARC: {0:0.4f}, LDLJ: {1:0.4f}'.format(sv, lv))

    ax = fig.add_subplot(235)
    plt.plot(a1[0], a1[1])
    plt.plot(a2[0], a2[1])
    ax.set_xlim(0, 50.)
    ax.set_xlabel('Frequency (hz)')
    ax.set_title('SPARC: {0:0.4f}, LDLJ: {1:0.4f}'.format(sa, la))

    ax = fig.add_subplot(236)
    plt.plot(j1[0], j1[1])
    plt.plot(j2[0], j2[1])
    ax.set_xlim(0, 50.)
    ax.set_xlabel('Frequency (hz)')
    ax.set_title('SPARC: {0:0.4f}, LDLJ: {1:0.4f}'.format(sj, lj))

    _str = 'Nvia: {0}, Nmove: {1}, File: {2}'
    plt.suptitle(_str.format(Nvia, Nmove, minfo['file']), fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9,
                        wspace=0.2, hspace=0.35)

    # Save plot as a PDF page
    pdf.savefig(fig)
    plt.close()


def analyse_smoothness_from_different_signals(data, fs, out_dir, diff_smooth):
    _cols = ('Nvia', 'Nmove', 'sparc_v', 'sparc_a', 'sparc_j',
             'ldlj_v', 'ldlj_a', 'ldlj_j')
    smoothness_summary = pd.DataFrame(columns=_cols)

    _outfile = "{0}/{1}/{2}".format(out_dir, diff_smooth['dir'],
                                    diff_smooth['fig_file'])
    with PdfPages(_outfile) as pdf:
        # Go through each data file and estimate smoothness using
        # both SPARC and LDLJ.
        for Nvia, Nmove, minfo, mdata in data:
            # Calculate speed, acceleration and jerk magnitude
            v = np.linalg.norm(np.array(mdata[['vx', 'vy', 'vz']]),
                               axis=1)[:-1]
            a = np.linalg.norm(np.array(mdata[['ax', 'ay', 'az']]),
                               axis=1)[:-2]
            j = np.linalg.norm(np.array(mdata[['jx', 'jy', 'jz']]),
                               axis=1)[:-3]

            # Estimate smoothness
            _sparcv, v1, v2 = sparc(v, fs=fs, fc=20., amp_th=0.05)
            _sparca, a1, a2 = sparc(a, fs=fs, fc=20., amp_th=0.05)
            _sparcj, j1, j2 = sparc(j, fs=fs, fc=20., amp_th=0.05)
            _ldljv = LDLJ(np.array(mdata[['vx', 'vy', 'vz']]), fs=fs,
                          data_type="vel", scale="ms")
            _ldlja = LDLJ(np.array(mdata[['ax', 'ay', 'az']]), fs=fs,
                          data_type="accl", scale="ms")
            _ldljj = LDLJ(np.array(mdata[['jx', 'jy', 'jz']]), fs=fs,
                          data_type="jerk", scale="ms")

            # Update data row
            _datarow = {'Nvia': [int(Nvia)], 'Nmove': [int(Nmove)],
                        'sparc_v': [_sparcv], 'sparc_a': [_sparca],
                        'sparc_j': [_sparcj], 'ldlj_v': [_ldljv],
                        'ldlj_a': [_ldlja], 'ldlj_j': [_ldljj]
                        }

            # Update smoothness summary DF
            smoothness_summary = pd.concat([smoothness_summary,
                                            pd.DataFrame.from_dict(_datarow)],
                                           ignore_index=True)

            # plot data
            generate_summary_plot_for_movement(pdf, mdata, minfo, Nvia, Nmove,
                                               fs, v, v1, v2, a, a1, a2, j,
                                               j1, j2, _sparcv, _sparca,
                                               _sparcj, _ldljv, _ldlja,
                                               _ldljj)

            sys.stdout.write("\r{0}".format(minfo['file']))
            sys.stdout.flush()

        # Update PDF file details
        d = pdf.infodict()
        d['Title'] = 'Smoothness estimates for different signals'
        d['Author'] = u'Sivakumar Balasubramanian'
        d['Subject'] = 'Smoothness Analysis'
        d['Keywords'] = 'Smoothness Analysis'
        d['CreationDate'] = datetime.datetime(2017, 12, 16)
        d['ModDate'] = datetime.datetime.today()

        # Save summary data
        _dfile = "{0}/{1}/{2}".format(out_dir, diff_smooth['dir'],
                                      diff_smooth['data_file'])
        smoothness_summary.to_csv(path_or_buf=_dfile, index=False)
        sys.stdout.write("\rDone!")


def summarize_sparc(diff_smooth):
    """Generates a summary plot comparing SPARC from different signals.
    """
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    plt.plot(diff_smooth['sparc_v'], diff_smooth['sparc_a'], 'o', alpha=0.7)
    ax.set_xlabel('SPARC Velocity', fontsize=15)
    ax.set_ylabel('SPARC Acceleration', fontsize=15)
    ax = fig.add_subplot(132)
    plt.plot(diff_smooth['sparc_v'], diff_smooth['sparc_j'], 'o', alpha=0.7)
    ax.set_xlabel('SPARC Velocity', fontsize=15)
    ax.set_ylabel('SPARC Jerk', fontsize=15)
    ax = fig.add_subplot(133)
    plt.plot(diff_smooth['sparc_a'], diff_smooth['sparc_j'], 'o', alpha=0.7)
    ax.set_xlabel('SPARC Accelertion', fontsize=15)
    ax.set_ylabel('SPARC Jerk', fontsize=15)
    plt.suptitle("SPARC from different signals", fontsize=20)
    plt.subplots_adjust(left=0.075, right=0.975, top=0.875, bottom=0.15,
                        wspace=0.3, hspace=0.35)
    return fig


def summarize_ldlj(diff_smooth):
    """Generates a summary plot comparing LDLJ from different signals.
    """
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    plt.plot(diff_smooth['ldlj_v'], diff_smooth['ldlj_a'], 'o', alpha=0.7)
    ax.set_xlabel('LDLJ Velocity', fontsize=15)
    ax.set_ylabel('LDLJ Acceleration', fontsize=15)
    ax = fig.add_subplot(132)
    plt.plot(diff_smooth['ldlj_v'], diff_smooth['ldlj_j'], 'o', alpha=0.7)
    ax.set_xlabel('LDLJ Velocity', fontsize=15)
    ax.set_ylabel('LDLJ Jerk', fontsize=15)
    ax = fig.add_subplot(133)
    plt.plot(diff_smooth['ldlj_a'], diff_smooth['ldlj_j'], 'o', alpha=0.7)
    ax.set_xlabel('LDLJ Accelertion', fontsize=15)
    ax.set_ylabel('LDLJ Jerk', fontsize=15)
    plt.suptitle("LDLJ from different signals", fontsize=20)
    plt.subplots_adjust(left=0.075, right=0.975, top=0.875, bottom=0.15,
                        wspace=0.3, hspace=0.35)
    return fig


def compare_sparc_ldlj(diff_smooth):
    """Generates a summary plot comparing SPARC and LDLJ measures for
    different signals."""
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    plt.plot(diff_smooth['sparc_v'], diff_smooth['ldlj_v'], 'o', alpha=0.7)
    ax.set_xlabel('SPARC Velocity', fontsize=15)
    ax.set_ylabel('LDLJ Velocity', fontsize=15)
    ax = fig.add_subplot(132)
    plt.plot(diff_smooth['sparc_a'], diff_smooth['ldlj_a'], 'o', alpha=0.7)
    ax.set_xlabel('SPARC Acceleration', fontsize=15)
    ax.set_ylabel('LDLJ Acceleration', fontsize=15)
    ax = fig.add_subplot(133)
    plt.plot(diff_smooth['sparc_j'], diff_smooth['ldlj_j'], 'o', alpha=0.7)
    ax.set_xlabel('SPARC Jerk', fontsize=15)
    ax.set_ylabel('LDLJ Jerk', fontsize=15)
    plt.suptitle("SPARC vs. LDLJ from different signals", fontsize=20)
    plt.subplots_adjust(left=0.075, right=0.975, top=0.875, bottom=0.15,
                        wspace=0.3, hspace=0.35)
    return fig


def compare_signals_measures(smooth_vals_max, smooth_vals_ms, dT, Ns):
    fig = plt.figure(figsize=(10.0, 6.0))
    # LDLJ versus inter-submovement interval for different 
    # submovement numbers from velocity, acceleration and jerk.
    cols = ['0.2', '0.4', '0.6']
    ax = fig.add_subplot(221)
    for i, _n in enumerate(Ns):
        ax.plot(dT, smooth_vals_max['vel'][i], lw=2, color=cols[i], label="$N_s={0}$".format(Ns[i]))
    ax.set_xticks([])
    ax.set_title('LDLJ Vel. (Max.)', fontsize=16)

    ax = fig.add_subplot(222)
    for i, _n in enumerate(Ns):
        ax.plot(dT, smooth_vals_ms['vel'][i], lw=2, color=cols[i], label="$N_s={0}$".format(Ns[i]))
    ax.set_xticks([])
    ax.set_title('LDLJ Vel. (MS)', fontsize=16)
    ax.legend(loc=3, prop={'size': 12}, handlelength=1.25, ncol=1, edgecolor='1.0', framealpha=0.0)

    ax = fig.add_subplot(223)
    for i, _n in enumerate(Ns):
        ax.plot(dT, smooth_vals_max['accl'][i], lw=2, color=cols[i], label="$N_s={0}$".format(Ns[i]))
    ax.set_xticks(np.arange(dT[0], dT[-1] + 0.5, 0.5))
    ax.set_xlabel("$\\Delta T$ (s)", fontsize=16)
    ax.set_title('LDLJ Accl. (Max.)', fontsize=16)

    ax = fig.add_subplot(224)
    for i, _n in enumerate(Ns):
        ax.plot(dT, smooth_vals_ms['accl'][i], lw=2, color=cols[i], label="$N_s={0}$".format(Ns[i]))
    ax.set_xticks(np.arange(dT[0], dT[-1] + 0.5, 0.5))
    ax.set_xlabel("$\\Delta T$ (s)", fontsize=16)
    ax.set_title('LDLJ Accl. (MS)', fontsize=16)

    plt.tight_layout()
    
    return fig


def compare_between_signals_and_measures(smooth_vals_max, smooth_vals_ms):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(221)
    ax.plot(smooth_vals_max['vel'][0], smooth_vals_ms['vel'][0], '.', color='0.4')
    ax.plot(smooth_vals_max['vel'][1], smooth_vals_ms['vel'][1], '.', color='0.4')
    ax.plot(smooth_vals_max['vel'][2], smooth_vals_ms['vel'][2], '.', color='0.4')
    ax.set_xlabel('LDLJ Vel. (Max.)', fontsize=16)
    ax.set_ylabel('LDLJ Vel. (MS)', fontsize=16)
    ax = fig.add_subplot(222)
    ax.plot(smooth_vals_max['accl'][0], smooth_vals_ms['accl'][0], '.', color='0.4')
    ax.plot(smooth_vals_max['accl'][1], smooth_vals_ms['accl'][1], '.', color='0.4')
    ax.plot(smooth_vals_max['accl'][2], smooth_vals_ms['accl'][2], '.', color='0.4')
    ax.set_xlabel('LDLJ Accl. (Max.)', fontsize=16)
    ax.set_ylabel('LDLJ Accl. (MS)', fontsize=16)
    ax = fig.add_subplot(223)
    ax.plot(smooth_vals_max['vel'][0], smooth_vals_max['accl'][0], '.', color='0.4')
    ax.plot(smooth_vals_max['vel'][1], smooth_vals_max['accl'][1], '.', color='0.4')
    ax.plot(smooth_vals_max['vel'][2], smooth_vals_max['accl'][2], '.', color='0.4')
    ax.set_xlabel('LDLJ Vel. (Max.)', fontsize=16)
    ax.set_ylabel('LDLJ Accl. (Max)', fontsize=16)
    ax = fig.add_subplot(224)
    ax.plot(smooth_vals_ms['vel'][0], smooth_vals_ms['accl'][0], '.', color='0.4')
    ax.plot(smooth_vals_ms['vel'][1], smooth_vals_ms['accl'][1], '.', color='0.4')
    ax.plot(smooth_vals_ms['vel'][2], smooth_vals_ms['accl'][2], '.', color='0.4')
    ax.set_xlabel('LDLJ Vel. (MS', fontsize=16)
    ax.set_ylabel('LDLJ Accl. (MS)', fontsize=16)
    plt.tight_layout()
    
    return fig


def compare_ldlj_vel_accl_ms(smooth_vals_ms, dT, Ns):
    colors = ['0', '0.35', '0.7']
    fig = plt.figure(figsize=(10, 3.5))

    # LDLJ versus inter-submovement interval for different 
    # submovement numbers from velocity, acceleration and jerk.
    ax = fig.add_subplot(121)
    for i, _n in enumerate(Ns):
        ax.plot(dT, smooth_vals_ms['vel'][i], lw=2.0, label="$N_s={0}$".format(Ns[i]), color=colors[i])
    ax.set_xticks(np.arange(dT[0], dT[-1] + 0.5, 0.5))
    ax.set_yticks(np.arange(-18, -3, 3));
    ax.set_xlabel("$\\Delta T$ (s)", fontsize=16)
    ax.set_title('LDLJ Vel.', fontsize=18)
    ax.legend(loc=1, prop={'size': 16}, handlelength=1.25, ncol=2, edgecolor='1.0', framealpha=0.0)

    ax = fig.add_subplot(122)
    for i, _n in enumerate(Ns):
        ax.plot(dT, smooth_vals_ms['accl'][i], lw=2.0, label="$N_s={0}$".format(Ns[i]), color=colors[i])
    ax.set_xticks(np.arange(dT[0], dT[-1] + 0.5, 0.5))
    ax.set_yticks(np.arange(-10, -2, 2));
    ax.set_xlabel("$\\Delta T$ (s)", fontsize=16)
    ax.set_title('LDLJ Accl.', fontsize=18)

    plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, wspace=0.2, hspace=0)

    return fig


def compare_ldlj_factors_val_accl_ms(smooth_vals_ms_fac, dT, Ns):
    colors = ['0', '0.6']
    fig = plt.figure(figsize=(10, 3.0))

    ax = fig.add_subplot(131)
    ax.plot(dT, smooth_vals_ms_fac['accl']['T'][0], lw=2.0, color=colors[1])
    ax.plot(dT, smooth_vals_ms_fac['vel']['T'][0], lw=2.0, color=colors[0])
    ax.set_xticks(np.arange(dT[0], dT[-1] + 0.5, 0.5))
    ax.set_xlabel("$\\Delta T$ (s)", fontsize=16)
    ax.set_title('- m x ln(T)', fontsize=16)

    ax = fig.add_subplot(132)
    ax.plot(dT, smooth_vals_ms_fac['accl']['A'][0], lw=2.0, color=colors[1])
    ax.plot(dT, smooth_vals_ms_fac['vel']['A'][0], lw=2.0, color=colors[0])
    ax.set_xticks(np.arange(dT[0], dT[-1] + 0.5, 0.5))
    ax.set_xlabel("$\\Delta T$ (s)", fontsize=16)
    ax.set_title('n x ln(A)', fontsize=16)

    ax = fig.add_subplot(133)
    ax.plot(dT, smooth_vals_ms_fac['accl']['J'][0], lw=2.0, color=colors[1], label='Accl')
    ax.plot(dT, smooth_vals_ms_fac['vel']['J'][0], lw=2.0, color=colors[0], label='Vel')
    ax.set_xticks(np.arange(dT[0], dT[-1] + 0.5, 0.5))
    ax.set_xlabel("$\\Delta T$ (s)", fontsize=16)
    ax.set_title('- ln(J)', fontsize=16)
    ax.legend(loc=1, prop={'size': 16}, handlelength=1.25, ncol=1, edgecolor='1.0', framealpha=0.0)

    plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, wspace=0.2, hspace=0)

    return fig


def ldlj_vel_accl_terms(ldlj_v, ldlj_v_facs, ldlj_a, ldlj_a_facs, dT):
    fig = plt.figure(figsize=(9, 6))
    # LDLJ-V vs. Inter. submovement interval
    ax = plt.subplot2grid((2, 6), (0, 0), colspan=3, rowspan=1)
    ax.plot(dT, ldlj_v[0, :], lw=2, label=f"2")
    ax.plot(dT, ldlj_v[1, :], lw=2, label=f"4")
    ax.plot(dT, ldlj_v[2, :], lw=2, label=f"8")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xlabel("$\delta T_i$ (sec)", fontsize=14)
    ax.set_title("LDLJ-V $\\left( \\lambda_L^v\\right)$", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)

    # LDLJ-A vs. Inter. submovement interval
    ax = plt.subplot2grid((2, 6), (0, 3), colspan=3, rowspan=1)
    ax.plot(dT, ldlj_a[0, :], lw=2, label=f"2")
    ax.plot(dT, ldlj_a[1, :], lw=2, label=f"4")
    ax.plot(dT, ldlj_a[2, :], lw=2, label=f"8")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xlabel("$\delta T_i$ (sec)", fontsize=14)
    ax.set_title("LDLJ-A $\\left( \\lambda_L^a\\right)$", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)

    ax = plt.subplot2grid((2, 6), (1, 0), colspan=2, rowspan=1)
    ax.plot(dT, ldlj_v_facs[0, :, 0], lw=2, label="$\\lambda_L^v$")
    ax.plot(dT, ldlj_a_facs[0, :, 0], lw=2, label="$\\lambda_L^a$")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xlabel("$\delta T_i$ (sec)", fontsize=14)
    ax.set_title("Duration", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(loc=1, frameon=False, fontsize=14)

    ax = plt.subplot2grid((2, 6), (1, 2), colspan=2, rowspan=1)
    ax.plot(dT, ldlj_v_facs[0, :, 1], lw=2, label="$\\lambda_L^v$")
    ax.plot(dT, ldlj_a_facs[0, :, 1], lw=2, label="$\\lambda_L^a$")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xlabel("$\delta T_i$ (sec)", fontsize=14)
    ax.set_title("Amplitude", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(loc=1, frameon=False, fontsize=14)

    ax = plt.subplot2grid((2, 6), (1, 4), colspan=2, rowspan=1)
    ax.plot(dT, ldlj_v_facs[0, :, 2], lw=2, label="$\\lambda_L^v$")
    ax.plot(dT, ldlj_a_facs[0, :, 2], lw=2, label="$\\lambda_L^a$")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xlabel("$\delta T_i$ (sec)", fontsize=14)
    ax.set_title("Jerk", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=1, frameon=False, fontsize=14)

    plt.tight_layout(pad=0.0, w_pad=0.25, h_pad=0.0)

    return fig


def sparc_vel_accl_terms(sparc_v, sparc_a, dT):
    fig = plt.figure(figsize=(9, 3))
    # SPARC on vel vs. Inter. submovement interval
    ax = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    ax.plot(dT, sparc_v[0, :], lw=2, label=f"2")
    ax.plot(dT, sparc_v[1, :], lw=2, label=f"4")
    ax.plot(dT, sparc_v[2, :], lw=2, label=f"8")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xlabel("$\delta T_i$ (sec)", fontsize=14)
    ax.set_title("SPARC on vel. $\\left( \\lambda_S^v\\right)$", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=3, frameon=False, fontsize=14, ncol=2)

    # SPARC on accl vs. Inter. submovement interval
    ax = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
    ax.plot(dT, sparc_a[0, :], lw=2, label=f"2")
    ax.plot(dT, sparc_a[1, :], lw=2, label=f"4")
    ax.plot(dT, sparc_a[2, :], lw=2, label=f"8")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xlabel("$\delta T_i$ (sec)", fontsize=14)
    ax.set_title("SPARC on accl. $\\left( \\lambda_S^v\\right)$", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(pad=0.0, w_pad=0.25, h_pad=0.0)

    return fig


def ldlj_vel_vs_accl(ldlj_v, ldlj_a):
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    # LDLJ-V vs. LDLJ-A
    _r1 = stats.pearsonr(ldlj_v, ldlj_a)[0]
    ax.plot(ldlj_v, ldlj_a, 'o', alpha=0.5, label=f"$\\rho = {_r1:0.3f}$")
    # ax.axis('equal')
    ax.set_xlim(-14.5, -5.5)
    ax.set_ylim(-6.6, -2.8)
    ax.set_xlabel("LDLJ-V $\\left(\\lambda_L^v\\right)$", fontsize=14)
    ax.set_ylabel("LDLJ-A $\\left(\\lambda_L^a\\right)$", fontsize=14)
    ax.set_title("LDLJ-V vs. LDLJ-A", fontsize=14)
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=4, frameon=False, fontsize=14)

    plt.tight_layout()

    return fig