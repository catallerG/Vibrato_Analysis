'''
# vibrato

Vibrato-specific functions.

Most noteworthy is find_vibrato_freq, which finds a single vibrato frequency for a given audio sample.
'''

import numpy as np
from scipy.signal import buttord, butter, lfilter
from acf_tracker import comp_acf, get_f0_from_acf, track_pitch_acf

from util import VibratoTrackerParams, block_audio, median_filter


def remove_mean(x):
    '''
    Normalize a signal to a mean of 0.

    Returns the signal and its mean as a tuple.
    '''

    mean = np.mean(x)
    normalized = x - np.mean(x)
    return normalized, mean


def vibrato_fs(fs, hop_size):
    '''
    Get the sample frequency of the f0 signal.

    # Parameters:

    fs: the sample frequency of the original audio
    hop_size (int): the hop size used when generating the original f0 signal
    '''

    return fs / hop_size


def vibrato_lpf(f0, acf_fs, f_cutoff=20):
    '''
    Apply a lowpass filter to an f0 signal to remove all audible frequencies.

    # Parameters

    f0: the signal to filter
    acf_fs: the sample frequency *of the f0 signal* (note this is NOT the sample frequency of the audio)
    f_cutoff: the cutoff frequency of the LPF in Hz (default 20 Hz)

    Best results are obtained if f0 is normalized to a mean of 0 first using remove_mean.
    '''

    # get the order and exact cutoff frequency for a butterworth filter with
    # < 3dB ripple in the passband and that attenuates >= 100dB in the stopband
    N, wn = buttord(f_cutoff * 0.9, f_cutoff * 1.1, 3, 100, fs=acf_fs)

    # get butterworth LPF coefficients
    b, a = butter(N, wn, fs=acf_fs)

    # apply the filter to f0 using the coefficients
    f0_filt = lfilter(b, a, f0)
    return f0_filt


def vibrato_frequency(f0, acf_fs, f0_is_normalized=False, interpolate=True):
    '''
    Find the vibrato frequency from the f0 signal.

    # Parameters:

    f0: the signal of fundamental frequencies
    acf_fs: the sample frequency *of the f0 signal* (note this is NOT the sample frequency of the audio)
    f0_is_normalized (boolean): set to True if f0 has had its mean removed using remove_mean
    interpolate (boolean): enable/disable interpolation for ACF

    # Returns:

    The computed vibrato frequency, or `NaN` if vibrato frequency cannot be computed
    '''

    if not f0_is_normalized:
        f0 = remove_mean(f0)[0]

    f0_r = comp_acf(f0)

    vibrato_f = get_f0_from_acf(f0_r, acf_fs, interpolate=interpolate)

    return vibrato_f


def compute_processed_f0(x, block_size, hop_size, fs, filter=True, interpolate=True):
    '''
    Compute f0 information for each block of x,
    then filter it and normalize to a mean of 0.

    x: the input audio signal
    block_size (int): the size of each processing block
    hop_size (int): the number of samples to increase for each successive block
    fs: sampling frequency
    filter (boolean): enable/disable filtering of fundamental frequency information
    interpolate (boolean): enable/disable interpolation for ACF

    Returns the normalized fundamental frequencies per-block and the sample
    frequency of these f0 values, as a tuple.
    '''

    f0 = track_pitch_acf(x, block_size, hop_size, fs, interpolate)[0]
    acf_fs = vibrato_fs(fs, hop_size)

    f0_norm = remove_mean(f0)[0]

    if filter:
        f0_norm = median_filter(f0_norm)
        f0_norm = vibrato_lpf(f0_norm, acf_fs)

    return f0_norm, acf_fs


def find_vibrato_freq_from_f0(f0_norm, acf_fs, interpolate=True):
    '''
    Find the vibrato frequency from f0 vector that has been normalized to a mean of 0.

    f0_norm: the f0 vector with zero mean
    acf_fs: the rate of each f0 sample
    interpolate (boolean): enable/disable interpolation for ACF

    '''

    vibrato_f = vibrato_frequency(f0_norm, acf_fs, True, interpolate)
    return vibrato_f


def find_vibrato_freq(x, block_size, hop_size, fs, filter=True, interpolate=True):
    '''
    Find the vibrato frequency for an audio signal.

    # Parameters:

    x: the input audio signal
    block_size (int): the size of each processing block
    hop_size (int): the number of samples to increase for each successive block
    fs: sampling frequency
    filter (boolean): enable/disable filtering of fundamental frequency information
    interpolate (boolean): enable/disable interpolation for ACF
    '''

    f0_norm, acf_fs = compute_processed_f0(
        x, block_size, hop_size, fs, filter, interpolate)

    vibrato_f = find_vibrato_freq_from_f0(f0_norm, acf_fs, interpolate)

    return vibrato_f


def track_vibrato_freq(x, params: VibratoTrackerParams):
    '''
    Track the vibrato frequency of a signal over time.

    x: the input audio signal
    params: see VibratoTrackerParams class
    '''

    block_size = params.block_size
    hop_size = params.hop_size
    fs = params.fs

    f0, acf_fs = compute_processed_f0(
        x, block_size, hop_size, fs, filter, params.interpolate)

    f0_window_size = params.window_size
    f0_hop_size = params.window_hop_size
    hop_denominator = params.hop_denominator

    # reuse the block_audio function to find our f0 windows
    windows, times = block_audio(f0, f0_window_size, f0_hop_size, acf_fs)

    # cut off the last few windows since the zero-padded windows are redundant and increase error
    cut_windows = hop_denominator - 1
    num_windows = windows.shape[0] - cut_windows

    times = times[:num_windows]
    windows = windows[:num_windows]

    # compute the f0 parameters per window
    params = np.zeros(num_windows)

    for i in range(num_windows):
        window = windows[i, :]
        vibrato_f = find_vibrato_freq_from_f0(window, acf_fs)

        params[i] = vibrato_f

    return params, times