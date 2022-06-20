# Copyright (c) neutrogic 2022

# references code from Python-WORLD and pyworld

import os
import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy import signal
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

np.set_printoptions(threshold=sys.maxsize)


global speed_dict
speed_dict = {0: 10.0, 1: 9.5, 2: 9.0, 3: 8.5, 4: 8.0,
              5: 7.5, 6: 7.0, 7: 6.5, 8: 6.0, 9: 5.5,
              10: 5.0, 11: 4.5, 12: 4.0, 13: 3.75, 14: 3.50,
              15: 3.25, 16: 3, 17: 2.75, 18: 2.5, 19: 2.25, 20: 2.0}


def world_initialize(in_wav):

    global path
    global wav
    global f0
    global sp
    global ap
    global fs
    global x
    global frame_val

    wav = in_wav
    frame_val = 5.0
    path = in_wav[:-4]

    print('---[WORLD]---\n\n[WORLD] Initializing World for ' + in_wav)  # Debug

    # Reads the wav data from tmp_combi and the frame size and makes it a WORLD wavform.
    x, fs = sf.read(os.path.join(Path.cwd(), 'o/tmp_combi.wav'))
    f0, sp, ap = pw.wav2world(x, fs)

    print('[WORLD] World has been itialized for ' + in_wav)  # Debug


def repitch(factor):

    global f0
    global wav

    print('[WORLD] Changing pitch of synthesis')  # Debug

    pitch_fact = factor
    if factor == 0:
        factor = 0.5
    factorr = float(factor / 10)

    f0 = f0 * factorr


def respeed(factor):

    global frame_val
    global speed_dict

    print('[WORLD] Changing speed of synthesis.')  # Debug

    frame_val = speed_dict.get(factor)

    print('[WORLD] Speed Changed')  # Debug


def breath(factor):

    # This isn't implemented yet cuz it sounds goofy

    global f0
    global x
    global fs
    global ap

    print('[WORLD] Changing breathiness of synthesis.')

    _f0, t = pw.harvest(x, fs)

    thresh = factor / 5

    ap = pw.d4c(x, f0, t, fs, threshold=thresh)


def expression(factor):

    global f0
    global x
    global fs
    global ap

    _f0, t = pw.harvest(x, fs)

    f02 = array(_f0)

    f0 = pw.stonemask(x, f02, t, fs)


def array(f0):

    a = np.float64(f0)
    a_list = list(a)
    mean = np.mean(a)
    b_list = []
    for i in range(len(a_list)):
        if a_list[i] == 0.0:
            b_list.append(0.0)
        else:
            b_list.append(a_list[i] / mean)

    c_list = []
    for i in range(len(a_list)):
        if b_list[i] == 0.0:
            c_list.append(0.0)
        else:
            c_list.append(b_list[i] * 0.75)

    d_list = []
    for i in range(len(a_list)):
        if c_list[i] == 0.0:
            d_list.append(0.0)
        else:
            if a_list[i] < mean:
                d_list.append(a_list[i] / c_list[i])
            elif a_list[i] > mean:
                d_list.append(a_list[i] * c_list[i])
            elif a_list[i] == mean:
                d_list.append(mean)

    f02 = np.asarray(d_list)
    print(a)
    print(f02)

    return f02


def world_write(pitch, speed, breath):

    global path
    global f0
    global sp
    global ap
    global fs
    global frame_val

    print('[WORLD] Synthesizing appended waveform')  # Debug

    resynth = pw.synthesize(f0, sp, ap, fs, frame_period=frame_val)

    path_new = path + '-resynth.wav'

    sf.write(path_new, resynth, fs)

    print('[WORLD] Completed resampling with WORLD')  # Debug
