'''
# util

General utility functions used across the project.
'''

import math
import numpy as np


def block_audio(x, block_size, hop_size, fs):
    '''
    Block the audio signal x into overlapping blocks.

    Returns matrix of blocks (dimensions `num_blocks` x `block_size`) and vector of
    start times for each block.

    ## Parameters:

    x: the audio signal
    block_size (int): the number of samples in each block
    hop_size (int): the number of samples to increase for each successive block
    fs: sample rate

    ## Note

    This function is adapted from Assignment 1 submission and reference solution.
    '''

    num_blocks = math.ceil(x.size / hop_size)

    # zero-pad x to have enough samples for the last block
    x = np.pad(x, (0, block_size))

    # pre-allocate the blocked audio
    xb = np.zeros((num_blocks, block_size))

    for n in range(num_blocks):
        start = n * hop_size
        # don't index out of range for the last block
        stop = np.min([x.size, start + block_size])

        xb[n] = x[start:stop]

    # compute time stamps
    t = np.arange(0, num_blocks) * hop_size / fs

    return xb, t


def quadratic_interp(a, b, c):
    '''
    Perform quadratic interpolation about the points (-1, a), (0, b), and (1, c).

    This function assumes that there is a local minimum or maximum between time indices -1 and 1.

    Returns the time index at which the extreme value is estimated to occur.

    Uses the formula given at the following page:
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    '''

    denominator = a - 2*b + c
    if denominator == 0:
        return 0

    return 0.5 * (a - c) / denominator


def generate_fm(fs, duration, f_carrier, f_modulator, depth):
    '''
    Generate a frequency-modulated sinusoid.

    Technically uses phase-modulation:
    x[t] = cos(2 * pi * f_carrier * t + depth * cos(2 * pi * f_modulator * t))

    # Parameters

    fs: sample frequency of the signal
    duration: duration of the signal, in seconds
    f_carrier: frequency of the carrier wave, in Hz
    f_modulator: frequency of the modulator wave, in Hz
    depth: depth of the modulation
    '''

    tt = np.arange(duration * fs) / fs
    mod = depth * np.cos(2 * np.pi * f_modulator * tt)
    x = np.cos(2 * np.pi * f_carrier * tt + mod)
    return x


def generate_am_fm(fs, duration, f_carrier, f_modulator, fm_depth, am_depth):
    '''
    Generate a sinusoid with amplitude- and frequency- modulation,
    where the modulators have the same frequency.

    Technically uses phase-modulation for the FM part:
    x[t] = cos(2 * pi * f_carrier * t + depth * cos(2 * pi * f_modulator * t))

    # Parameters

    fs: sample frequency of the signal
    duration: duration of the signal, in seconds
    f_carrier: frequency of the carrier wave, in Hz
    f_modulator: frequency of the modulator waves, in Hz
    fm_depth: depth of the frequency modulation
    am_depth: depth of the amplitude modulation
    '''

    tt = np.arange(duration * fs) / fs
    modulator = np.cos(2 * np.pi * f_modulator * tt)
    fm = fm_depth * modulator
    am = am_depth * modulator
    x = am * np.cos(2 * np.pi * f_carrier * tt + fm)
    return x


def median_filter(x, n=3):
    '''
    Apply a median filter to a signal.

    As the name suggests, a median filter selects the median of the values in a window of a signal.
    Median filtering is effective at removing sudden outliers from a signal.
    Median filtering does not remove large bursts of outliers.

    # Parameters

    x: the input signal
    n (optional), int: the number of points to consider per window

    Returns a new signal, median filtered.
    '''

    from scipy.signal import medfilt
    filtered = medfilt(x, n)
    return filtered


DEFAULT_FILTER = True
DEFAULT_INTERPOLATE = True
DEFAULT_HOP_DENOM = 5
DEFAULT_WINDOW_DUR = 0.5


class VibratoTrackerParams:
    '''
    A utility class to hold vibrato tracker parameters and allow for default parameters.
    '''

    def __init__(self, block_size, hop_size, fs, **kwargs):
        '''
        Parameters:

        block_size
        hop_size
        fs
        filter (optional, bool)
        interpolate (optional, bool)
        hop_denominator (optional)
        window_duration (optional)
        '''

        from vibrato import vibrato_fs
        from math import ceil

        self.block_size = block_size
        self.hop_size = hop_size
        self.fs = fs

        self.filter = kwargs.get('filter', DEFAULT_FILTER)
        self.interpolate = kwargs.get('interpolate', DEFAULT_INTERPOLATE)
        self.hop_denominator = kwargs.get('hop_denominator', DEFAULT_HOP_DENOM)
        self.window_duration = kwargs.get(
            'window_duration', DEFAULT_WINDOW_DUR)

        self.acf_fs = vibrato_fs(self.fs, self.hop_size)
        self.window_size = ceil(self.acf_fs * self.window_duration)
        self.window_hop_size = ceil(self.window_size / self.hop_denominator)
