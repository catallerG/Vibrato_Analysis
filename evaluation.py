import glob
import numpy as np
import csv
from vibrato import track_vibrato_freq
from util import VibratoTrackerParams
from rms_tracker import track_rms_vibrato_freq
from vib_identifier import vib_identifier
import librosa


def open_dataset(filepath):
    fh = open(filepath, 'rt')
    annotation = []
    try:
        reader = csv.reader(fh)
        for row in reader:
            annotation.append(row)
    except Exception as e:
        print("Exception is:", e)
    finally:
        fh.close()
    annotation = np.array(annotation)
    return annotation


def readFile(filepath):
    #rat, wav = scipy.io.wavfile.read(filepath)
    wav, rat =librosa.load(filepath, sr=None)
    duratio = len(wav) / rat
    tim = np.arange(0, duratio, 1 / rat)
    return rat, wav, duratio, tim


def if_down_mix(wav):
    if wav.shape[-1] != 2:
        return wav
    else:
        mono = wav[:, 0] / 2 + wav[:, 1] / 2
        return mono


def open_files(filepath, block_size, hop_size, fs):
    result = np.zeros((0, 2))
    time_sequence = np.zeros(0)
    folder = glob.glob(filepath + "\\*.wav")
    for count, wavFile in enumerate(folder):
        fs, wave, duration, timeinsec = readFile(wavFile)
        wave = if_down_mix(wave)
        InputParams = VibratoTrackerParams(block_size, hop_size, fs)
        params, times = track_vibrato_freq(wave, InputParams)
        #params, times = track_rms_vibrato_freq(wave, InputParams)
        filename = folder[count].split("\\")[len(folder[count].split("\\"))-1]
        filenames = np.array([filename] * np.shape(params)[0])
        vib_status = vib_identifier(wave) #sequence of "vibrato on" or "vibrato off"
        data = np.vstack((params, filenames))
        #data = np.vstack((data, vib_status))
        result = np.vstack((result, data.T))
        time_sequence = np.hstack((time_sequence, times))
    return result, time_sequence


def compute_vib_status_overlap_ratio(result, time_sequence, annotation):
    error_time = 0
    time = time_sequence[np.shape(time_sequence)[1]-1]
    for i, row in enumerate(annotation):
        if row[2] == "no vibrato" or row[9] == "nan":
            continue
        start_time = float(np.split(row[9], ":")[0])
        end_time = float(np.split(row[9], ":")[1])
        for n in np.where(result[:, 1]==row[1])[0]: #same file name
            if n+1 < np.shape(time_sequence)[1]:
                if result[n][2] == "vibrato off" and time_sequence[n] >= start_time and time_sequence[n+1] < end_time:
                    error_time = error_time + time_sequence[n+1] - time_sequence[n]
    return 1-error_time/time


def compute_vib_rate_accuracy(result, time_sequence, annotation):
    accuracy_for_each_vib = []
    for i, row in enumerate(annotation):
        if i == 0:
            continue
        if row[2] == "no vibrato" or row[9] == "nan":
            continue
        start_time = float(row[9].split(":")[0])
        end_time = float(row[9].split(":")[1])
        deviation_one_vibrato = []
        for n in np.where(result[:, 1]==row[1])[0]: #same file name
            if n+1 < np.shape(time_sequence)[0]:
                #if result[n][2] == "vibrato on" and time_sequence[n] >= start_time and time_sequence[n+1] < end_time:
                if time_sequence[n] >= start_time and time_sequence[n + 1] < end_time:
                    if float(result[n][0]) > 0:
                        deviation = abs(float(result[n][0])-float(row[5]))
                        deviation_one_vibrato.append(deviation)
        if len(deviation_one_vibrato)>0:
            average_deviation = sum(deviation_one_vibrato)/len(deviation_one_vibrato)
            accuracy_for_each_vib.append(average_deviation/float(row[5]))
    accuracy_for_each_vib = np.array(accuracy_for_each_vib)
    accuracy_for_each_vib = accuracy_for_each_vib * 100
    return accuracy_for_each_vib #actually it's a percent deviation


annotation = open_dataset("D:\SchoolWork\ACA/vibrato_analysis-master\Dataset\MTG-Violin/violin_vibrato(fixed).csv")
result, time_sequence = open_files("D:\SchoolWork\ACA/vibrato_analysis-master\Dataset\MTG-Violin", 2048, 1024, 44100)
accuracy_array = compute_vib_rate_accuracy(result, time_sequence, annotation)
print(accuracy_array)
print(np.mean(accuracy_array))
