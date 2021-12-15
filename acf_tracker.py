'''
# ACF Tracker

Functions related to tracking the fundamental frequency of an audio signal using
the autocorrelation function (ACF).

**Note:** much of this module is adapted from Assignment 1 submission and
reference solution.
'''

import numpy as np

from util import block_audio, quadratic_interp


def comp_acf(x, normalize=True):
    '''
    Compute the right half of the autocorrelation function of a signal.

    ## Parameters:

    x: the audio signal
    normalize (boolean): if `True`, the output of the ACF is normalized by the magnitude of x
    '''

    if normalize:
        norm = np.dot(x, x)
    else:
        norm = 1

    correlation = np.correlate(x, x, "full")

    # only return the right half of the ACF
    correlation = correlation[x.size-1:correlation.size]

    return correlation / norm


def get_f0_from_acf(r, fs, interpolate=False):
    '''
    Estimate the fundamental frequency of a block, given the right side of
    its autocorrelation function.

    ## Parameters:

    r: the block's autocorrelation function
    fs: the sample rate of the block
    interpolate (boolean): use quadratic interpolation for better f0 estimate resolution

    ## Returns:

    the computed f0 in Hz, or 0 if there is no discernible peak in the ACF
    '''

    cutoff = 1

    # ignore the first peak at n = 0
    # find the next peak afterwards by looking past the first minimum value
    diffs = np.diff(r)

    # handle edge case when there are no diffs > 0
    # in this case, there's no peak, so there isn't any reasonable f0 to report
    positive_diffs = np.where(diffs > 0)[0]
    if positive_diffs.size == 0:
        return 0
    else:
        cutoff_tmp = positive_diffs[0]

    cutoff = max(cutoff, cutoff_tmp)

    # find the max value now
    f = np.argmax(r[cutoff+1:r.size]) + cutoff + 1

    if interpolate and f > 0:
        a = r[f-1]
        b = r[f]

        if f+1 == r.size:
            c = r[f]
        else:
            c = r[f+1]

        adjustment = quadratic_interp(a, b, c)

        f += adjustment

    return fs / f


def track_pitch_acf(x, block_size, hop_size, fs, interpolate=False):
    '''
    Track the fundamental frequency of an audio signal over time.

    Returns two arrays: fundamental frequency and time.

    ## Parameters:

    x: the audio signal
    block_size (int): the size of each processing block
    hop_size (int): the number of samples to increase for each successive block
    fs: sampling frequency in Hz
    interpolate (boolean): use quadratic interpolation for better f0 estimate resolution
    '''

    xb, t = block_audio(x, block_size, hop_size, fs)
    num_blocks = xb.shape[0]

    # resulting fundamental frequencies
    f0 = np.zeros(num_blocks)

    for n in range(num_blocks):
        block = xb[n, :]
        r = comp_acf(block)
        f0[n] = get_f0_from_acf(r, fs, interpolate)

    return f0, t