
"""
The module contains classes and functions to generate different types of
simulated movements.

Author: Sivakumar Balasubramanian
Date: March 13, 2014
"""

import numpy as np


def data_span(data):
    """
    Returns the ampltide span of the given list or array.

    Parameters
    ----------
    data     : np.array, list, tuple
               The data whose amplitude span is to be calcuated.

    Returns
    -------
    span     : float
               The span, defined as the different between the max. and min.
               values of the given data.

    Notes
    -----

    Examples
    --------
    >>> data_span([-1,4,0,-23,27,0.573])
    50
    >>> data_span([])
    Empty list or array provided! Span cannot be calculated.
    """

    if len(data) == 0:
        print("Empty list or array provided! Span cannot be calculated.")
        return

    return max(data) - min(data)


def mjt_discrete_movement(amp=1.0, dur=1.0, loc=0.,
                          time=np.arange(-0.5, 0.5, 0.01),
                          data_type='vel'):
    """
    Generate a discrete Minumum Jerk Trajectory (MJT) movement speed profile
    of the given amplitude, duration and time location for the given time span.

    Parameters
    ----------
    amp         : float
                  Amplitude of the MJT discrete movement.
    dur         : float
                  Duration of the MJT discrete movement.
    loc         : float
                  The temporal location of the center of the MJT speed
                  profile.
    time        : np.array
                  The time values for which the speed profile values are to be
                  returned.
    data_type   : string
                  The type data returned by the function. The function can 
                  return velocity, acceleration or jerk. There are only three 
                  possibiliies, {'vel', 'accl', 'jerk'}, corresponding to 
                  velocity, acceleration and jerk.
    Returns
    -------
    movement    : np.array
                  The movement speed profile of the MJT discrete movement.

    Notes
    -----

    Examples
    --------
    """
    # make sure data_type makes sense.
    if data_type not in ('vel', 'accl', 'jerk'):
        _str = '\n'.join(("data_type has to be ('vel', 'accl', 'jerk')!",
                          "{0} provided is not valid".format(data_type)))
        raise Exception(_str)
        return
    
    _t = (time + 0.5 * dur - loc) / dur
    t = np.array([np.min([np.max([_tt, 0.]), 1.]) for _tt in _t])
    if data_type == 'vel':
        return amp * np.polyval([30, -60, 30, 0, 0], t) / np.power(dur, 1)
    elif data_type == 'accl':
        return amp * np.polyval([120, -180, 60, 0], t) / np.power(dur, 2)
    else:
        _w = 1.0 * (_t >= 0) * (_t <= 1)
        return amp * (np.polyval([360, -360, 60], t) * _w / np.power(dur, 3))


def gaussian_discrete_movement(amp=1.0, dur=1.0, loc=0.,
                               time=np.arange(-0.5, 0.5, 0.01),
                               data_type='vel'):
    """
    Generate a discrete Gaussian movement speed profile of the given amplitude,
    duration and time location for the given time span.

    Parameters
    ----------
    amp         : float
                  Amplitude of the Gaussian discrete movement.
    dur         : float
                  Duration of the Gaussian discrete movement.
    loc         : float
                  The temporal location of the center of the Gaussian speed
                  profile.
    time        : np.array
                  The time values for which the speed profile values are to be
                  returned.
    data_type   : string
                  The type data returned by the function. The function can 
                  return velocity, acceleration or jerk. There are only three 
                  possibiliies, {'vel', 'accl', 'jerk'}, corresponding to 
                  velocity, acceleration and jerk.
    Returns
    -------
    movement    : np.array
                  The movement speed profile of the Gaussian discrete movement.

    Notes
    -----

    Examples
    --------
    """
    # make sure data_type makes sense.
    if data_type not in ('vel', 'accl', 'jerk'):
        _str = '\n'.join(("data_type has to be ('vel', 'accl', 'jerk')!",
                          "{0} provided is not valid".format(data_type)))
        raise Exception(_str)
        return
    
    _t = 5.0 * (time - loc) / dur
    _tt = 5.0 / dur
    _v = amp * np.exp(-pow(_t, 2))
    if data_type == 'vel':
        return _v
    elif data_type == 'accl':
        return _v * (-2 * _t) * _tt
    else:
        return _v * (np.power(-2 * _t, 2) * np.power(_tt, 2) +
                     (-2 * np.power(_tt, 2)))


def gaussian_rhytmic_movement(amp, dur, interval, ts, n_movements):
    """
    Generates a rhythmic (repetitive) Gaussian movement speed profiles that
    is the sum of individual Gaussian speed profiles.

    Parameters
    ----------
    amp         : float
                  Amplitud of each submovement.
    dur         : float
                  Duration of each submovement.
    interval    : float
                  Time interval between the peaks two successive Gaussian
                  movements.
    ts          : float

    n_movements : int
                  The number of Gaussian movements in the overall movementsnt.
    Returns
    -------
    time        : np.array
                  The time of the movement starting from 0.
    movement    : np.array
                  The movement speed profile of the Gaussian rhythmic movement.

    Notes
    -----

    """
    time = np.arange(0., (n_movements - 1) * interval + dur, ts)
    movement = np.zeros(len(time))
    for i in range(n_movements):
        movement += gaussian_discrete_movement(amp, dur,
                                               loc=i * interval + 0.5 * dur,
                                               time=time)
    return time, movement


def generate_random_movement(move_type='gaussian'):
    """
    Generates a random movement as a sum of Gaussian submovements. The number
    of submovements, their duration, time location and amplitude are chosen
    randomly.

    Parameters
    ----------
    move_type       : string
                      This must be a string indicating the type of discrete
                      movement to use for generating the random movement.

    Returns
    -------
    t               : np.array
                      The time of the movement starting from 0.
    movement        : np.array
                      The speed profile of the generated random movement.
    submovements    : np.array
                      A two dimensional array containing the individual
                      submovements in the generated random movement. The number
                      of row is equal to the number of submovements, and the
                      number of columns corresponds to the number of data
                      points for the duration of the generated movement.

    Notes
    -----

    """
    t_sample = 0.01

    # get a random set of parameters for the movement.
    Ns = np.random.randint(1, 10, 1)[0]  # number of submovements
    As = 0.8 * np.random.rand(Ns) + 0.2  # amplitude of submovements
    Ts = 0.3 * np.random.rand(Ns) + 0.3  # duration of submovements
    T0 = 0.5 * np.random.rand(Ns)  # location of submovements
    tmin = T0[0] - max(Ts) / 2
    tmax = sum(T0) + max(Ts) / 2
    t = np.arange(tmin, tmax, t_sample)
    movement = np.zeros((tmax - tmin + t_sample) / t_sample)
    submovements = [0] * Ns
    # movement function
    move_func = (gaussian_discrete_movement if move_type.lower() == 'gaussian'
                 else mjt_discrete_movement)
    for i in range(Ns):
        submovements[i] = move_func(As[i], Ts[i], sum(T0[:i + 1]), t)
        movement += submovements[i]
    return t, movement, np.array(submovements)


def generate_movement(Ns, amp, dT, T, ts=0.001, move_type='gaussian',
                      data_type='vel'):
    """
    Generates a movement as sum of submovements with the given parameters.

    Parameters
    ----------
    Ns              : int
                      This indicates the number of submovements
    amp             : np.array
                      The amplitude of the Ns submovements.
    dT              : np.array
                      This is the inter-submovement interval. This is
                      of length Ns-1, as it only contains the intervals
                      with repsect to the previous submovement. The
                      first submovement is assumed to start from zero,
                      i.e. the its center is located at half its duration.
    T               : np.array
                      The durations of the Ns submovements.
    ts              : float
                      This is the sampling duration.
    move_type       : string
                      This must be a string indicating the type of
                      submovement to use - 'gaussian' or 'mjt'.
    data_type       : string
                      The type data returned by the function. The function can 
                      return velocity, acceleration or jerk. There are only three 
                      possibiliies, {'vel', 'accl', 'jerk'}, corresponding to 
                      velocity, acceleration and jerk.

    Returns
    -------
    t               : np.array
                      The time of the movement starting from 0.
    movement        : np.array
                      The speed profile of the generated random movement.
    submovements    : np.array
                      A two dimensional array containing the individual
                      submovements in the generated random movement. The number
                      of row is equal to the number of submovements, and the
                      number of columns corresponds to the number of data
                      points for the duration of the generated movement.

    Notes
    -----

    """
    # get movement end time.
    tmax = get_movement_endtime(dT, T)
    t = np.arange(0., tmax, ts)
    # initialize movement and submovement variables.
    movement = np.zeros(len(t))
    submovements = [0] * Ns
    # get movement function
    move_func = (gaussian_discrete_movement if move_type.lower() == 'gaussian'
                 else mjt_discrete_movement)
    for i in range(Ns):
        submovements[i] = move_func(amp[i], T[i], sum(dT[:i]) + 0.5 * T[i],
                                    t, data_type)
        movement += submovements[i]
    return t, movement, np.array(submovements)


def get_movement_endtime(dT, T):
    """
    Returns the end time of the movement assuming that the start time is zero.
    """
    _t = 0.
    for i, _dT in enumerate(T):
        _t = np.max([_t, np.sum(dT[:i]) + T[i]])
    return _t
