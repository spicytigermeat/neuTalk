import csv
import gc
import os
import sys
import tarfile
from pathlib import Path
from tkinter import *
from tkinter.filedialog import askopenfilename

import ttkbootstrap as ttk
from ControllableTalkNet.core.extract import ExtractDuration
from ftfy import fix_text as fxy
from talknet import load_tacotron_model, load_talknet_models, load_pipeline_model
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

PATH = Path(__file__).parent

# Center's window or toplevel


def center(root, w, h):
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)

    return ('%dx%d+%d+%d' % (w, h, x, y))

# Verifies an int and prevents entry of a non-int


def verify_int(event):
    v = event.char
    try:
        v = int(v)
    except ValueError:
        if v != '\x08' and v != '':
            return 'break'

# Verifies a float and prevents entry of non-float


def verify_float(event):
    v = event.char
    try:
        v = float(v)
    except ValueError:
        if v != '\x08' and v != '':
            return 'break'

# Does nothing.


def donashi():
    return


def lib_fmt_tool(lang):

    import speakerlibrarytool as libtool

    libtool.main_window(lang)

    del libtool
    gc.collect()


def get_defaults(spkr_dir, pause_length, pho_dict, normalize):

    default_path = PATH / 'default_param.csv'
    default_dict = {}

    with open(default_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        default_dict = {fxy(rows[0]): fxy(rows[1]) for rows in reader}

    spkr_dir.set(default_dict['speaker'])
    pause_length.set(default_dict['pause_length'])
    pho_dict.set(os.path.join(PATH, 'dicts', default_dict['dict']))
    normalize.set(default_dict['normalize'])

    ExtractDuration._set_dictionary(pho_dict.get())

    return spkr_dir, pause_length, normalize, default_dict


def lang_init(default_dict):
    lang_list = []
    lang_dir = 'gui_langs'

    with open(os.path.join(lang_dir, default_dict['lang']), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lang = {rows[0]: rows[1] for rows in reader}

    lang_list = os.listdir('gui_langs')

    lang_menu = ['English (US)', 'Japanese']

    return lang_menu, lang_list, lang


def read_lang_lists(dir):

    with open(os.path.join(pro_dir, 'gui_langs', dir), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        temp_dict = {rows[0]: rows[1] for rows in reader}

    return temp_dict['LANG']


def get_csv_dict(spkr_dir):
    cfg_dict = {}
    with open(os.path.join(spkr_dir, 'speaker.ntk_cfg'), 'r+', encoding='utf-8') as f:
        reader = csv.reader(f)
        cfg_dict = {rows[0]: rows[1] for rows in reader}
        return cfg_dict


def speaker_list(spkrs_dir):
    spkr_dir = [f.path for f in os.scandir(spkrs_dir) if f.is_dir()]
    return spkr_dir


def menu_list(speaker):
    menu = []
    for i in range(len(speaker)):
        with open(os.path.join(speaker[i], 'speaker.ntk_cfg'), 'r+', encoding='utf-8') as f:
            lines = fxy(f.read())
            first = lines.split('\n', 1)[0]
            name = first.split(',', 1)[1]
            menu.append(name)
    return menu


def load_spkr(spkr_dir, sub_index):

    spkr_dict = get_csv_dict(spkr_dir)

    sub_spkr_dir_list = []
    sub_spkr_menu_list = []

    model_list = {k: v for k, v in spkr_dict.items() if k.startswith('_')}

    if len(model_list) == 1:
        sub_spkr_dir_list.append(os.path.join(
            spkr_dir, 'models', model_list['_m0'].split('|')[0]))
        sub_spkr_menu_list.append(model_list['_m0'].split('|')[-1])
        if spkr_dict['lang'] == 'eng':
            load_talknet_models(sub_spkr_dir_list[0])
        elif spkr_dict['lang'] == 'jpn':
            load_tacotron_model(sub_spkr_dir_list[0])
        elif spkr_dict['lang'] == 'zh':
            load_tacotron_model(sub_spkr_dir_list[0])
        elif spkr_dict['lang'] == 'fr':
            load_tacotron_model(sub_spkr_dir_list[0])
        elif spkr_dict['lang'] == 'eng_tt2':
            load_tacotron_model(sub_spkr_dir_list[0])
        elif spkr_dict['lang'] == 'eng_tt2_arpa':
            load_tacotron_model(sub_spkr_dir_list[0])
        elif spkr_dict['lang'] == 'pipeline':
            load_pipeline_model(sub_spkr_dir_list[0])
    else:
        for i in range(len(model_list)):
            sub_spkr_dir_list.append(os.path.join(
                spkr_dir, 'models', model_list['_m' + str(i)].split('|')[0]))
            sub_spkr_menu_list.append(model_list['_m' + str(i)].split('|')[-1])
        if spkr_dict['lang'] == 'eng':
            load_talknet_models(sub_spkr_dir_list[sub_index])
        elif spkr_dict['lang'] == 'jpn':
            load_tacotron_model(sub_spkr_dir_list[sub_index])
        elif spkr_dict['lang'] == 'zh':
            load_tacotron_model(sub_spkr_dir_list[sub_index])
        elif spkr_dict['lang'] == 'fr':
            load_tacotron_model(sub_spkr_dir_list[sub_index])
        elif spkr_dict['lang'] == 'eng_tt2':
            load_tacotron_model(sub_spkr_dir_list[sub_index])
        elif spkr_dict['lang'] == 'eng_tt2_arpa':
            load_tacotron_model(sub_spkr_dir_list[sub_index])
        elif spkr_dict['lang'] == 'pipeline':
            load_pipeline_model(sub_spkr_dir_list[sub_index])

    return spkr_dict, sub_spkr_dir_list, sub_spkr_menu_list


def install_spkr(root, lang):

    type = [
        ('NTKPKG File', '*.ntkpkg')
    ]

    try:
        ntkpkg = askopenfilename(
            parent=root, title=fxy(lang['Select a speaker in .ntkpkg format.']), filetypes=type,
            defaultextension=type)

        if not os.path.exists(os.path.join('speakers', os.path.basename(ntkpkg[:-7]))):
            os.mkdir(os.path.join('speakers', os.path.basename(ntkpkg[:-7])))

        with tarfile.open(ntkpkg, 'r') as tar:
            tar.extractall(os.path.join(
                'speakers', os.path.basename(ntkpkg[:-7])))
            tar.close()

        Messagebox.ok(
            title=fxy(lang['neuTalk']),
            message=fxy(lang['Speaker successfully installed.'] + '\n' +
                        lang['Please restart neuTalk.'])
        )
    except OSError:
        Messagebox.ok(
            title=fxy(lang['Error']),
            message=fxy(lang['Incorrect file type.'] + '\n' +
                        lang['Please select a .ntkpkg file.']
                        ))

    return ntkpkg
