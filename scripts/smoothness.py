
"""
smoothness.py contains a list of functions for estimating movement smoothness.
"""
import numpy as np


def sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = sparc(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'

    """
    try:
        # Number of zeros to be padded.
        nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

        # Frequency
        f = np.arange(0, fs, fs / nfft)
        # Normalized magnitude spectrum
        Mf = abs(np.fft.fft(movement, nfft))
        Mf = Mf / max(Mf)

        # Indices to choose only the spectrum within the given cut off
        # frequency Fc.
        # NOTE: This is a low pass filtering operation to get rid of high
        # frequency noise from affecting the next step (amplitude threshold
        # based cut off for  arc length calculation).
        fc_inx = ((f <= fc) * 1).nonzero()
        f_sel = f[fc_inx]
        Mf_sel = Mf[fc_inx]

        # Choose the amplitude threshold based cut off frequency.
        # Index of the last point on the magnitude spectrum that is greater
        # than or equal to the amplitude threshold.
        inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
        fc_inx = range(inx[0], inx[-1] + 1)
        f_sel = f_sel[fc_inx]
        Mf_sel = Mf_sel[fc_inx]

        # Calculate arc length
        new_sal = -sum(
            np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                    pow(np.diff(Mf_sel), 2)))
        return new_sal, (f, Mf), (f_sel, Mf_sel)
    except:
        return np.NaN, np.NaN, np.NaN


def dimensionless_jerk_factors(movement, fs, data_type='vel', rem_mean=False):
    """
    Returns the individual factors of the dimensionless jerk metric.

    Parameters
    ----------
    movement    : np.array
                  The array containing the movement velocity (acceleration or
                  jerk) profile. This can be multi-dimensional with the rows
                  corresponding to the time samples and the columns
                  corresponding to the different dimensions.
    fs          : float
                  The sampling frequency of the data.
    data_type   : string
                  The type of movement data provided. This will determine the
                  scaling factor to be used. There are only three possibiliies,
                  {'vel', 'accl'}, corresponding to velocity, and acceleration.
    rem_mean    : booleans
                  This indicates if the mean of the given movement data must be
                  removed before comupting the jerk. It must be noted that when
                  the movement data is velocity, this parameter is ignored.
                  This parameter is used only when the movement data is 
                  acceleration or jerk.

    Returns
    -------
    T^N      : float
               Duration scaling factor.
    A^M      : float
               Amplitude scaling factor.
    J        : float
               Jerk cost.

    Notes
    -----


    Examples
    --------
    """
    # Pamameter definition for different data types
    param = {'vel': {'n': 2, 'N': 3},
             'accl': {'n': 1, 'N': 1}}

    n, N = (param[data_type]['n'], param[data_type]['N'])

    # make sure data_type makes sense.
    if data_type not in ('vel', 'accl'):
        _str = '\n'.join(("data_type has to be ('vel', 'accl')!",
                          "{0} provided is not valid".format(data_type)))
        raise Exception(_str)
        return
    
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

    # Remove the mean if the movement dat is acceleration?
    if data_type == 'accl' and rem_mean == True:
        movement = movement - np.mean(movement, axis=0)

    # jerk
    jerk = np.linalg.norm(np.diff(movement, axis=0, n=n), axis=1)
    jerk /= np.power(dt, n)
    mjerk = np.sum(np.power(jerk, 2)) * dt

    # time.
    _N = len(movement)
    mdur = np.power(_N * dt, N)

    # amplitude.
    mamp = np.power(np.max(np.linalg.norm(movement, axis=1)), 2)

    # dlj factors
    return mdur, mamp, mjerk


def dimensionless_jerk(movement, fs, data_type='vel', rem_mean=False):
    """
    Calculates the smoothness metric for the given velocity profile using the
    dimensionless jerk metric.

    Parameters
    ----------
    movement    : np.array
                  The array containing the movement velocity (acceleration or
                  jerk) profile. This can be multi-dimensional with the rows
                  corresponding to the time samples and the columns
                  corresponding to the different dimensions.
    fs          : float
                  The sampling frequency of the data.
    data_type   : string
                  The type of movement data provided. This will determine the
                  scaling factor to be used. There are only three possibiliies,
                  {'vel', 'accl'}, corresponding to velocity, and acceleration.
    rem_mean    : booleans
                  This indicates if the mean of the given movement data must be
                  removed before comupting the jerk. It must be noted that when
                  the movement data is velocity, this parameter is ignored.
                  This parameter is used only when the movement data is 
                  acceleration or jerk.

    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.

    Notes
    -----


    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(np.array([move]).T, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'
    >>> dl = dimensionless_jerk(np.array([move, move, move]).T, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'

    """
    # Factors for dimensionless jer
    dljfac = dimensionless_jerk_factors(movement, fs, data_type, rem_mean)

    # estimate dj
    return - (dljfac[0] / dljfac[1]) * dljfac[2]


def log_dimensionless_jerk_factors(movement, fs, data_type='vel',
                                   rem_mean=False):
    """
    Returns the individual factors of the dimensionless jerk metric.

    Parameters
    ----------
    movement    : np.array
                  The array containing the movement velocity (acceleration or
                  jerk) profile. This can be multi-dimensional with the rows
                  corresponding to the time samples and the columns
                  corresponding to the different dimensions.
    fs          : float
                  The sampling frequency of the data.
    data_type   : string
                  The type of movement data provided. This will determine the
                  scaling factor to be used. There are only three possibiliies,
                  {'vel', 'accl'}, corresponding to velocity, and acceleration.
    rem_mean    : booleans
                  This indicates if the mean of the given movement data must be
                  removed before comupting the jerk. It must be noted that when
                  the movement data is velocity, this parameter is ignored.
                  This parameter is used only when the movement data is 
                  acceleration or jerk.
    Returns
    -------
    -ln(T^N) : float
               Duration scaling factor.
    +ln(A^M) : float
               Amplitude scaling factor.
    -ln(J)   : float
               Jerk cost.

    Notes
    -----


    Examples
    --------
    """
    dljfac = dimensionless_jerk_factors(movement, fs, data_type, rem_mean)
    return - np.log(dljfac[0]), np.log(dljfac[1]), - np.log(dljfac[2])


def log_dimensionless_jerk(movement, fs, data_type='vel',
                           rem_mean=False):
    """
    Calculates the smoothness metric for the given movement velocity,
    acceleration or jerk profile using the log dimensionless jerk metric.

    Parameters
    ----------
    movement    : np.array
                  The array containing the movement velocity (acceleration or
                  jerk) profile. This can be multi-dimensional with the rows
                  corresponding to the time samples and the columns
                  corresponding to the different dimensions.
    fs          : float
                  The sampling frequency of the data.
    data_type   : string
                  The type of movement data provided. This will determine the
                  scaling factor to be used. There are only three possibiliies,
                  {'vel', 'accl'}, corresponding to velocity, and acceleration.
    rem_mean    : booleans
                  This indicates if the mean of the given movement data must be
                  removed before comupting the jerk. It must be noted that when
                  the movement data is velocity, this parameter is ignored.
                  This parameter is used only when the movement data is 
                  acceleration or jerk.

    Returns
    -------
    ldlj      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.

    Notes
    -----


    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = log_dimensionless_jerk(np.array([move]).T, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'
    >>> dl = log_dimensionless_jerk(np.array([move, move, move]).T, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'

    """
    ldljfac = log_dimensionless_jerk_factors(movement, fs, data_type, rem_mean)
    return ldljfac[0] + ldljfac[1] + ldljfac[2]


def log_dimensionless_jerk_imu_factors(accls, gyros, grav, fs):
    """
    Returns the individual factors of the log dimensionless jerk metric
    used for IMU data.

    Parameters
    ----------
    accls : np.array
            The array containing the accelerometer profile. This is a
            multi-dimensional with the rows corresponding to the time samples
            and the columns corresponding to the x, y, and z components.
    gyros : np.array
            The array containing the gyroscope profile. This is 
            multi-dimensional with the rows corresponding to the time samples
            and the columns corresponding to the x, y, and z components.
    grav  : np.array
            Gravity vector. This is a 3x1 array with the x, y, and z component
            of gravity. 
    fs    : float
            The sampling frequency of the data.

    Returns
    -------
    -ln(T) : float
               Duration scaling factor.
    +ln(A) : float
               Amplitude scaling factor.
    -ln(J)   : float
               Jerk cost.

    Notes
    -----


    Examples
    --------
    """
    # Sample time
    dt = 1. / fs
    _N = len(accls)

    # Movement duration.
    mdur = _N * dt

    # Gravity subtracted mean square ampitude
    mamp = np.power(np.linalg.norm(accls), 2) / _N
    if gyros is not None:
      mamp = mamp - np.power(np.linalg.norm(grav), 2)
    
    # Derivative of the accelerometer signal
    _daccls = np.vstack((np.zeros((1, 3)), np.diff(accls, axis=0) * fs)).T
    
    # Get corrected jerk if gyroscope data is available.
    if gyros is not None:
      _awcross = np.array([np.cross(_as, _ws)
                           for _as, _ws in zip(accls, gyros)]).T
    else:
      _awcross = np.zeros(np.shape(_daccls))
    
    # Corrected jerk
    _jsc = _daccls - _awcross
    mjerk = np.sum(np.power(np.linalg.norm(_jsc, axis=0), 2)) * dt

    return - np.log(mdur), np.log(mamp), - np.log(mjerk)


def log_dimensionless_jerk_imu(accls, gyros, grav, fs):
    """
    Calculates the smoothness metric for the given IMU data, accelerometer
    and gyroscope signals, using the log dimensionless jerk metric.

    Parameters
    ----------
    accls : np.array
            The array containing the accelerometer profile. This is a
            multi-dimensional with the rows corresponding to the time samples
            and the columns corresponding to the x, y, and z components.
    gyros : np.array
            The array containing the gyroscope profile. This is 
            multi-dimensional with the rows corresponding to the time samples
            and the columns corresponding to the x, y, and z components.
    grav  : np.array
            Gravity vector. This is a 3x1 array with the x, y, and z component
            of gravity. 
    fs    : float
            The sampling frequency of the data.

    Returns
    -------
    ldlj  : float
            The log dimensionless jerk estimate of the given movement's
            smoothness.

    Notes
    -----


    Examples
    --------

    """
    # Get factors for the LDLJ calculation
    _f = log_dimensionless_jerk_imu_factors(accls, gyros, grav, fs)
    return _f[0] + _f[1] + _f[2]


def sgr(accls, grav):
    """
    Returns the siganl-to-gravity ratio for the given acceleration signal.

    Parameters
    ----------
    accls : np.array
            The array containing the accelerometer profile. This is a
            multi-dimensional with the rows corresponding to the time samples
            and the columns corresponding to the x, y, and z components.
    grav  : float
            The value of gravity. Please ensure that the value is in the same
            units as the accls signal.

    Returns
    -------
    sgr   : float
            Signal-to-gravity ratio

    Notes
    -----


    Examples
    --------
    """
    # first enforce data into an numpy array.
    movement = np.array(movement)
    r, c = np.shape(movement)

    dt = 1. / fs

    return np.linalg.norm(accls) / (np.sqrt(r) * grav)
