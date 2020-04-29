
"""
This module contains the a list of supporting classes and functions for the
notebook investigating smoothness estimates directly from different types of
signals.

Author: Sivakumar Balasubramanian
Date: 16 Dec 2017
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import sys
sys.path.append("scripts/")
from smoothness import sparc
from smoothness import log_dimensionless_jerk


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
            _ldljv = log_dimensionless_jerk(
                np.array(mdata[['vx', 'vy', 'vz']]), fs=fs, data_type="vel", scale="ms")
            _ldlja = log_dimensionless_jerk(
                np.array(mdata[['ax', 'ay', 'az']]), fs=fs, data_type="accl", scale="ms")
            _ldljj = log_dimensionless_jerk(
                np.array(mdata[['jx', 'jy', 'jz']]), fs=fs, data_type="jerk", scale="ms")

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


class PdfPlotWriter:
    """Class to handle saving a series of files to a single
    PDF file.
    """
    
    def __init__(self, filename, title="A PDF File", author="Some Author"):
        self._pdf = PdfPages(filename)
        # Write metadata
        d = self._pdf.infodict()
        d['Title'] = title
        d['Author'] = author
        d['CreationDate'] = datetime.datetime.today()
        
    def add_fig(self, fig, note=""):
        """Add the given figure as a page to the current PDF file.
        """
        if fig is not None:
            self._pdf.attach_note(note)
            self._pdf.savefig(fig)
            plt.close(fig)
        else:
            self._pdf.close()