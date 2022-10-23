import csv
import globalparam

import os
import sys
import time
import argparse
import resampy
import uuid
import time
from scipy.io.wavfile import write
from pydub import AudioSegment, effects
import os.path
import numpy as np
import sys
import soundfile as sf
import torch, json, emoji
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams as hps
from uberduck_ml_dev.vocoders.hifigan import HiFiGanGenerator
from uberduck_ml_dev.models.tacotron2 import Tacotron2 as tt2
from uberduck_ml_dev.models.tacotron2 import DEFAULTS
from uberduck_ml_dev.data_loader import prepare_input_sequence, pad_sequences
from uberduck_ml_dev.data_loader import text_to_sequence as tts
from uberduck_ml_dev.models.torchmoji import TorchMojiInterface
import numpy
import soundfile as sf
import gc
import warnings
import base64
import time
from uberduck_ml_dev.text.symbols import (
    DEFAULT_SYMBOLS,
    NVIDIA_TACO2_SYMBOLS,
)



import gc
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import warnings
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from ControllableTalkNet.core import extract
from nemo.collections.tts.models.base import (SpectrogramGenerator,
                                              TextToWaveform, Vocoder)
#from nemo.collections.tts.models.hifigan import HifiGanModel
from nemo.collections.tts.models.talknet import (TalkNetDursModel,
                                                 TalkNetPitchModel,
                                                 TalkNetSpectModel)
from ntk_g2p import jpn, zh, fr, eng_tt2, eng_tt2_arpa, pipeline
from omegaconf import OmegaConf, open_dict
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib
import matplotlib.pylab as plt

sys.path.append('tacotron2')
from model import Tacotron2
from hparams import hparam_dict as hdict
from text import text_to_sequence
sys.path.append('hifigan')
from models import Generator
from denoiser import Denoiser
from env import AttrDict
from meldataset import MAX_WAV_VALUE

warnings.filterwarnings('ignore')


sys.path.append('/')

logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)

# This code was adapted from the inference notebook
# included in NeMo and the one by SortAnon for
# Controllable talknet.
# It's been edited to only
# include talknet as the spectrogram model and
# hifigan as the vocoder model, and to load the
# model similarly to how i did it with tacotron2 implementation.


global DEVICE
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

RUN_PATH = os.path.dirname(os.path.realpath(__file__))

extract_dur = extract.ExtractDuration(Path.cwd(), DEVICE)


def load_talknet_models(modeldir):

    global tn_model
    global vocoder_model
    global spec_gen
    global hifigan
    global h
    global denoiser

    if not os.path.exists('temp'):
        os.mkdir('temp')

    with tarfile.open(modeldir, 'r:') as tar:
        tar.extractall('temp')
        tar.close()

    hifigan_path = 'temp/hifiganmodel'
    spect_path = 'temp/TalkNetSpect.nemo'
    durs_path = 'temp/TalkNetDurs.nemo'
    pitch_path = 'temp/TalkNetPitch.nemo'
    f0_path = 'temp/f0_info.json'
    config = 'temp/config.json'

    unload_models()

    # Load TalkNet Models
    with torch.no_grad():

        tn_model = TalkNetSpectModel.restore_from(spect_path)

        tn_dur = TalkNetDursModel.restore_from(durs_path)
        tn_model.add_module('_durs_model', tn_dur)

        tn_pit = TalkNetPitchModel.restore_from(pitch_path)
        tn_model.add_module('_pitch_model', tn_pit)

        if torch.cuda.is_available() == True:
            spec_gen = tn_model.eval().cuda()
        else:
            spec_gen = tn_model.eval()

        load_hifigan_model(hifigan_path, config)
        if torch.cuda.is_available() == True:
            hifigan.eval().cuda()
        else:
            hifigan.eval()
        hifigan.remove_weight_norm()

    if os.path.exists('temp'):
        try:
            shutil.rmtree('temp')
        except OSError as e:
            print('Error %s : %s' % 'temp', e.strerror)


def load_tacotron_model(modeldir):

    global vocoder_model, hifigan, h, denoiser, tt_model

    unload_models()

    if not os.path.exists('temp'):
        os.mkdir('temp')

    with tarfile.open(modeldir, 'r:') as tar:
        tar.extractall('temp')
        tar.close()

    hifigan_path = 'temp/hifiganmodel'
    config = 'temp/config.json'
    tacotron_model = 'temp/tacotron2.pt'

    hparams = hdict()

    if torch.cuda.is_available() == True:
        tt_model = Tacotron2(hparams).cuda()
        if hparams['fp16_run']:
            tt_model.decoder.attention_layer.score_mask_value = finfo(
                'float16').min

        if hparams['distributed_run']:
            tt_model = apply_gradient_allreduce(tt_model)
        tt_model.load_state_dict(torch.load(tacotron_model)['state_dict'])

        _ = tt_model.cuda().eval().half()
    else:
        tt_model = Tacotron2(hparams)
        if hparams['fp16_run']:
            tt_model.decoder.attention_layer.score_mask_value = finfo(
                'float16').min

        if hparams['distributed_run']:
            tt_model = apply_gradient_allreduce(tt_model)

        tt_model.load_state_dict(torch.load(
            tacotron_model, map_location=torch.device('cpu'))['state_dict'])
        _ = tt_model.eval()

    hifigan = load_hifigan_model(hifigan_path, config)
    if torch.cuda.is_available() == True:
        hifigan.eval().cuda()
    else:
        hifigan.eval()
    hifigan.remove_weight_norm()

def load_pipeline_model(modeldir):
    
    unload_models()

    global filename
    global symbol_set
    global cpu_run
    global torchmoji
    global use_gpu
    global taco
    global hifigan
    global speakerid

    if not os.path.exists('temp'):
        os.mkdir('temp')

    with tarfile.open(modeldir, 'r:') as tar:
        tar.extractall('temp')
        tar.close()

    warnings.filterwarnings("ignore")

    EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
    :pensive: :ok_hand: :blush: :heart: :smirk: \
    :grin: :notes: :flushed: :100: :sleeping: \
    :relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
    :sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
    :neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
    :v: :sunglasses: :rage: :thumbsup: :cry: \
    :sleepy: :yum: :triumph: :hand: :mask: \
    :clap: :eyes: :gun: :persevere: :smiling_imp: \
    :sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
    :wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
    :angry: :no_good: :muscle: :facepunch: :purple_heart: \
    :sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')

    synthesis_length = 20
    gate_threshold = 0.05
    default_config = False
    use_torchmoji = True
    modelo = 'temp/hifiganmodel'
    if torch.cuda.is_available() == True:
        use_gpu = True
    else:
        use_gpu = False
    cpu_run = (not use_gpu)
    device = "cpu" if not use_gpu else "cuda"
    gc.enable()
    filename = None
    nome_modelo = 'temp/tacotron2.pt'
    torchmoji = TorchMojiInterface(
                    "vocabulary.json",
                    "pytorch_model.bin",
    )
    config = DEFAULTS.values()
    modelo = 'temp/hifiganmodel'
    hifigan = HiFiGanGenerator(
    config="temp/config.json",
    checkpoint=modelo,
    cudnn_enabled=use_gpu
    )
    model = torch.load(nome_modelo, map_location=device)
    symbol_set = "nvidia_taco2"
    cleaner = 'english_cleaners'
    modelo = 'temp/hifiganmodel'
    fn = "temp/conf.json"
    with open(fn) as f:
        config.update(json.load(f))
    config.update(
          {
            "max_decoder_steps": synthesis_length * 100,
            "gate_threshold": gate_threshold,
            "symbol_set": symbol_set
          }
    )
    hparams = hps(**config)
    taco = tt2(hparams)
    taco.from_pretrained(warm_start_path=nome_modelo, device=device)

    with open(modeldir + '/../../speaker.ntk_cfg', 'r') as file:
            reader = csv.reader(file)
            row1 = next(reader)
            row2 = next(reader)
            row3 = next(reader)
            row4 = next(reader)
            row5 = next(reader)
            row6 = next(reader)
            row7 = next(reader)
            row8 = next(reader)
            speakerid = row8[1]


def load_hifigan_model(hifigan_path, config):

    global hifigan
    global h
    global denoiser
    global DEVICE

    # Load HifiGan Models
    with open(config, encoding='utf-8') as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device(DEVICE))
    state_dict_g = torch.load(
        (hifigan_path), map_location=torch.device(DEVICE))
    hifigan.load_state_dict(state_dict_g['generator'])
    denoiser = Denoiser(hifigan, mode='normal')
    
    return hifigan


def unload_models():

    global spec_gen, hifigan, h, denoiser, tt_model
    spec_gen, hifigan, h, denoiser, tt_model = '', '', '', '', ''
    del spec_gen, hifigan, h, denoiser, tt_model
    gc.collect()


def infer(str_input):

    global hifigan
    global h
    global denoiser
    global spec_gen

    with torch.no_grad():

        token_list, tokens, arpa = extract_dur.get_tokens(str_input)

        spect = spec_gen.generate_spectrogram(tokens=tokens)

        y_g_hat = hifigan(spect.float())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del audio_np, token_list, tokens, arpa, spect, y_g_hat, audio, audio_denoised
    gc.collect()


def infer_f0_shift(text):

    global hifigan
    global h
    global denoiser
    global spec_gen

    with torch.no_grad():

        token_list, tokens, arpa = extract_dur.get_tokens(text)

        durs = extract_dur.get_duration(wav, text, token_list)

        f0_with_silence, f0s_wo_silence = extract_pitch.get_pitch()

        print(f0_with_silence)

        spect = spec_gen.force_spectrogram(tokens=tokens,
                                           durs=torch.from_numpy(durs).view(
                                               1, -1).type(torch.LongTensor).to(DEVICE),
                                           f0=torch.FloatTensor(f0_with_silence).view(1, -1).to(DEVICE))

        y_g_hat = hifigan(spect.float())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del audio_np, token_list, tokens, arpa, spect, y_g_hat, audio, audio_denoised
    gc.collect()
    
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')

def infer_jpn(str_input):

    global hifigan
    global h
    global denoiser
    global tt_model

    with torch.no_grad():

        sequence = np.array(text_to_sequence(
            jpn.get_phones(str_input), ['convert_to_ascii']))[None, :]
        
        print(jpn.get_phones(str_input))
        """
        if torch.cuda.is_available() == True:
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
        else:
        """
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).long()

        mel, mel_post, _, alignment = tt_model.inference(sequence)

        y_g_hat = hifigan(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del audio_np, sequence, mel, mel_post, _, alignment, y_g_hat, audio, audio_denoised
    gc.collect()

def infer_pipeline(str_input):
    with torch.no_grad():
        text = str_input
        if not filename:
            start = time.time()
            arpabet = True
            arpabet = 0.0 if not arpabet else 1.0
            speakerID = int(speakerid)

        if text.find("|")!=-1:
            text = text.split("|")
            torchmoji_override = text[1]
            text = text[0]
        else:
            torchmoji_override = ""
        cleaner = 'english_cleaners'
        seqs = []
        seqs.append(
            torch.IntTensor(
                tts(
                    text=text,
                    cleaner_names=[cleaner],
                    p_arpabet=arpabet,
                    symbol_set=symbol_set,
                )[:]
            )
        )

        compute_gst = lambda texts: torchmoji.encode_texts(texts)
        text_padded, input_lengths = pad_sequences(seqs)
        if not cpu_run:
            text_padded = text_padded.cuda().long()
            input_lengths = input_lengths.cuda().long()
        else:
            text_padded = text_padded.long()
            input_lengths = input_lengths.long()

        embedding = None
        if torchmoji_override != "":
            embedding = compute_gst([torchmoji_override])
        else:
            embedding = compute_gst([text])
        embedding = torch.FloatTensor(embedding)

        emojis = torchmoji.enc2emojis(embedding)[0]
        emojo = (emoji.emojize(f" {' '.join(emojis)}", language="alias"))

        embedding = embedding.cuda() if use_gpu else embedding
        speakerembedding =  torch.LongTensor([speakerID]).cuda() if use_gpu else torch.LongTensor([speakerID])
        input_ = [text_padded, input_lengths, speakerembedding, embedding]
        output = taco.inference(input_)
        audio = hifigan.infer(output[1][:1])
        audio_np = audio

    return audio_np

def infer_zh(str_input):

    global hifigan
    global h
    global denoiser
    global tt_model

    with torch.no_grad():

        sequence = np.array(text_to_sequence(
            zh.get_phones(str_input), ['return_text']))[None, :]
        
        print(zh.get_phones(str_input))
        """
        if torch.cuda.is_available() == True:
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
        else:
        """
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).long()

        mel, mel_post, _, alignment = tt_model.inference(sequence)

        y_g_hat = hifigan(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del audio_np, sequence, mel, mel_post, _, alignment, y_g_hat, audio, audio_denoised
    gc.collect()

def infer_fr(str_input):

    global hifigan
    global h
    global denoiser
    global tt_model

    with torch.no_grad():
        
        sequence = np.array(text_to_sequence(
            fr.get_phones(str_input), ['return_text']))[None, :]
        
        print(fr.get_phones(str_input))
        """
        if torch.cuda.is_available() == True:
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
        else:
        """
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).long()

        mel, mel_post, _, alignment = tt_model.inference(sequence)

        y_g_hat = hifigan(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del audio_np, sequence, mel, mel_post, _, alignment, y_g_hat, audio, audio_denoised
    gc.collect()

def infer_eng_tt2(str_input):

    global hifigan
    global h
    global denoiser
    global tt_model

    with torch.no_grad():

        sequence = np.array(text_to_sequence(
            eng_tt2.get_phones(str_input), ['english_cleaners']))[None, :]
        
        print(eng_tt2.get_phones(str_input))
        """
        if torch.cuda.is_available() == True:
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
        else:
        """
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).long()

        mel, mel_post, _, alignment = tt_model.inference(sequence)

        y_g_hat = hifigan(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del audio_np, sequence, mel, mel_post, _, alignment, y_g_hat, audio, audio_denoised
    gc.collect()

def infer_eng_tt2_arpa(str_input):

    global hifigan
    global h
    global denoiser
    global tt_model

    with torch.no_grad():

        sequence = np.array(text_to_sequence(
            eng_tt2_arpa.get_phones(str_input), ['english_cleaners']))[None, :]
        
        print(eng_tt2_arpa.get_phones(str_input))
        """
        if torch.cuda.is_available() == True:
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
        else:
        """
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).long()

        mel, mel_post, _, alignment = tt_model.inference(sequence)

        y_g_hat = hifigan(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del audio_np, sequence, mel, mel_post, _, alignment, y_g_hat, audio, audio_denoised
    gc.collect()

def synthesize(text):

    audio = infer(text)

    return audio

    del audio
    gc.collect()


def synthesize_jpn(text):

    audio = infer_jpn(text)

    return audio

    del audio
    gc.collect()

def synthesize_zh(text):

    audio = infer_zh(text)

    return audio

    del audio
    gc.collect()

def synthesize_fr(text):

    audio = infer_fr(text)

    return audio

    del audio
    gc.collect()
    
def synthesize_eng_tt2(text):

    audio = infer_eng_tt2(text)

    return audio

    del audio
    gc.collect()

def synthesize_eng_tt2_arpa(text):

    audio = infer_eng_tt2_arpa(text)

    return audio

    del audio
    gc.collect()

def synthesize_pipeline(text):

    audio = infer_pipeline(text)

    return audio

    del audio
    gc.collect()


def wav_transfer(text, pitch_fact, wav):

    global hifigan
    global h
    global denoiser
    global spec_gen

    with torch.no_grad():

        print('[DEBUG] Transfering Wav')

        extract_dur = extract.ExtractDuration(RUN_PATH, DEVICE)
        extract_pitch = extract.ExtractPitch()

        new_wav = wav + '_conv.wav'

        ffmpeg.input(wav).output(
            new_wav,
            ar='22050',
            ac='1',
            acodec='pcm_s16le',
            map_metadata='-1',
            fflags='+bitexact'
        ).overwrite_output().run(quiet=True)

        token_list, tokens, arpa = extract_dur.get_tokens(text)

        durs = extract_dur.get_duration(wav, text, token_list)

        f0_with_silence, f0s_wo_silence = extract_pitch.get_pitch(new_wav)

        spect = spec_gen.force_spectrogram(tokens=tokens,
                                           durs=torch.from_numpy(durs).view(
                                               1, -1).type(torch.LongTensor).to(DEVICE),
                                           f0=torch.FloatTensor(f0_with_silence).view(1, -1).to(DEVICE))

        y_g_hat = hifigan(spect.float())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)

    return audio_np

    del extract_dur, extract_pitch, new_wav, token_list, tokens, arpa, durs, f0_with_silence, f0s_wo_silence
    del y_g_hat, spect, audio, audio_denoised, audio_np
    gc.collect()
