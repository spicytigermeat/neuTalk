import re
from pyopenjtalk import g2p
from pypinyin import pinyin, lazy_pinyin, Style
from phonemizer import phonemize
from phonemizer.separator import Separator
from g2p_en import G2p
from ftfy import fix_text as fxy
import tkinter.filedialog as fd
import shutil
import os

from phonemizer.backend.espeak.wrapper import EspeakWrapper

class eng(): #including this so I can edit english input outside of the tokenizer from talknet

    def __init__():
        super().__init__()

    def get_fixes(input_text):

        output_text = input_text + '.'

        return output_text

class jpn():
    
    def __init__():
        super().__init__()
    
    def get_phones(input_text):
        
        output_text = ''
     
        output_text = re.sub('pau', '.', g2p(input_text))
        output_text = re.sub('cl', 'Q', output_text)
        output_text = re.sub('\s', '', output_text) + '.'

        return output_text

class zh():

    def __init():
        super().__init__()

    def get_phones(input_text):

        output_list = []

        output_list = lazy_pinyin(fxy(input_text), style=Style.TONE3)

        output_text = ''

        for i in range(len(output_list)):
            output_text = output_text + output_list[i] + ' '

        output_text = fxy(re.sub('。', '.', output_text) + '.')

        return output_text

class fr():
    def __init__():
        super().__init__()

    def get_phones(input_text):
        EspeakWrapper.set_library('libespeak-ng.dll')
        output_text = ''
        output_text = phonemize(
            output_text,
            language='fr-fr',
            backend='espeak',
            separator=Separator(phone=None, word=' ', syllable=None),
            strip=True,
            preserve_punctuation=True,
            njobs=4)
        return output_text

class eng_tt2():
    def __init__():
        super().__init__()

    def get_phones(input_text):
        output_text = input_text
        return output_text

class eng_tt2_arpa():
    def __init__():
        super().__init__()

    def get_phones(input_text):
        g2p = G2p()
        output_text = g2p(input_text)
        output_text = '{' + ' '.join(output_text) + '}'
        output_text = fxy(re.sub('   ', '} {', output_text))
        output_text = fxy(re.sub('{.}', '.', output_text))
        return output_text
