# Copyright (c) neutrogic 2022

# Parses text for tt2/hfg inference and stitches audio together. not sure why i put them in a seperate script but i'm not going to apologize for it.

import os
import re
import sys
from pathlib import Path

from ntk_g2p import eng
from pydub import AudioSegment
from pydub.effects import normalize

global o_dir
o_dir = os.path.join(Path.cwd(), 'o')

global vol_dict
vol_dict = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1,
            5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5}

# Cut text into a list


def text_cutter(text):

    lines = []
    lines = text.replace(',', '.'
                         ).replace('!', '.'
                                   ).replace('?', '.'
                                             ).replace('\n', ''
                                                       ).replace('。', '.'
                                                                 ).replace('、', '.'
                                                                           ).replace('！', '.'
                                                                                     ).replace('？', '.'
                                                                                               ).replace(']', ']]'
                                                                                                         ).split('.')

    for i in range(len(lines)):
        if lines[i] == '':
            lines.pop(i)

    for i in range(len(lines)):
        lines[i] = lines[i] + '.'

    print(lines)

    return lines
#

# Stitch audio files together and deletes unneeded temp files.


def audio_stitch(pause_len, volume_factor, normalize):

    global o_dir
    global vol_dict

    pau = AudioSegment.silent(duration=pause_len)
    files = os.listdir(os.path.join(o_dir))

    sound = AudioSegment.silent(duration=100)

    for file in files:
        sound = sound + AudioSegment.from_wav(os.path.join(o_dir, file)) + pau

    for f in os.listdir(o_dir):
        os.remove(os.path.join(o_dir, f))

    if normalize == 'yes':
        sound = sound.normalize()

    if volume_factor != 5:

        fact = vol_dict.get(volume_factor)

        print(fact)

        if volume_factor > 5:
            sound = sound + fact
        if volume_factor < 5:
            sound = sound - fact

    sound.export(os.path.join(o_dir, 'tmp_combi.wav'),
                 format='wav', bitrate='64k')
