'''
# RMS tracker
'''

import numpy as np

from util import VibratoTrackerParams, block_audio
from vibrato import vibrato_fs


def block_rms(x):
    '''
    Get the RMS of the signal x.

    The minimum value returned is -100 dB (10^-5).

    Returns the RMS in the same units as the input, *NOT* decibels.
    '''

    rms = np.sqrt(np.dot(x, x) / x.size)
    epsilon = 1e-5
    if rms < epsilon:
        rms = epsilon

    return rms


def extract_rms(xb):
    '''
    Track the RMS of blocked audio.

    Returns the RMS in the same units as the input, *NOT* decibels.

    Returns a vector of RMS values with length equal to the number of blocks in xb.
    '''

    num_blocks = xb.shape[0]

    rms_values = np.zeros(num_blocks)

    for i in range(num_blocks):
        rms = block_rms(xb[i, :])
        rms_values[i] = rms

    return rms_values


def track_rms(x, block_size, hop_size, fs):
    '''
    Track the RMS energy of an audio signal over time.

    Returns two arrays: RMS and time (in seconds).

    ## Parameters:

    x: the audio signal
    block_size (int): the size of each processing block
    hop_size (int): the number of samples to increase for each successive block
    fs: sampling frequency in Hz
    '''

    xb, t = block_audio(x, block_size, hop_size, fs)
    rms = extract_rms(xb)
    return rms, t


def find_vibrato_freq_from_rms(rms, acf_fs, interpolate=True):
    # reuse vibrato_frequency function from vibrato.py
    from vibrato import vibrato_frequency

    vibrato_f = vibrato_frequency(rms, acf_fs, False, interpolate)

    # crucially, this is *twice* the frequency of the signal
    vibrato_f /= 2

    return vibrato_f


def track_rms_vibrato_freq(x, params: VibratoTrackerParams):
    block_size = params.block_size
    hop_size = params.hop_size
    fs = params.fs

    rms, t = track_rms(x, block_size, hop_size, fs)
    acf_fs = vibrato_fs(fs, hop_size)

    rms_window_size = params.window_size
    rms_hop_size = params.window_hop_size
    hop_denominator = params.hop_denominator

    # reuse the block_audio function to find our rms value windows
    windows, times = block_audio(rms, rms_window_size, rms_hop_size, acf_fs)

    # cut off the last few windows since the zero-padded windows are redundant and increase error
    cut_windows = hop_denominator - 1
    num_windows = windows.shape[0] - cut_windows

    times = times[:num_windows]
    windows = windows[:num_windows]

    # compute the f0 parameters per window
    vibrato_freqs = np.zeros(num_windows)

    for i in range(num_windows):
        window = windows[i, :]
        vibrato_f = find_vibrato_freq_from_rms(window, acf_fs)

        vibrato_freqs[i] = vibrato_f

    return vibrato_freqs, times