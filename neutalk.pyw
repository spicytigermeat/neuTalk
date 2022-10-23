# =========================================================
# ===== neuTalk: Text-to-Speech GUI Tool by neutrogic =====
# =========================================================
# === Copyright (c) neutrogic 2022 ===

import csv
import gc
import importlib
import itertools
import math
import os
import re
import sys
import tarfile
import time
import webbrowser
from pathlib import Path
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename

import globalparam
import simpleaudio as sa
import talknet
import ttkbootstrap as ttkB
import world_neutalk
from ControllableTalkNet.core.extract import ExtractDuration
from ftfy import fix_text as fxy
from globalparam import speaker_cfg, syn_param
#from ntk_util import dictionary_tool
from ntk_util import center, donashi, get_defaults, install_spkr, lang_init
from ntk_util import lib_fmt_tool as libtool
from ntk_util import (load_spkr, menu_list, speaker_list, verify_float,
                      verify_int)
from PIL import Image, ImageTk
from pydub import AudioSegment
#import librosa
from scipy.io.wavfile import write
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.themes import user

# Easy to call directory
global pro_dir
pro_dir = Path.cwd()

global neutalk_ver
neutalk_ver = '0.6'

global spkr_dir
global spkr_name
global spkr_img
global spkr_author
global spkr_models
global spkr_ver
global spkr_readme
global pause_length
global syn_pitch
global syn_speed
global syn_volume
global init_spkr

def change_default_lang(set_win):

    global curr_lang
    global default_dict
    global lang

    default_dict['lang'] = fxy(curr_lang.get())

    try:
        with open('default_param.csv', 'w', encoding='utf-8') as f:
            for key in default_dict.keys():
                f.write('%s,%s\n' % (key, fxy(default_dict[key])))
    except IOError:
        print('I/O Error')

    Messagebox.ok(
        title=fxy(lang['Default Language']),
        message=fxy(lang['Change will appear upon reopening neuTalk.'])
    )

    set_win.destroy()

def update_default(event):

    global default_dict

    default_dict['normalize'] = event.get()
    default_dict['lang'] = fxy(curr_lang.get())

    try:
        with open('default_param.csv', 'w', encoding='utf-8') as f:
            for key in default_dict.keys():
                f.write('%s,%s\n' % (key, fxy(default_dict[key])))
    except IOError:
        print('I/O Error')


def export_txt():

    global t_in

    filetypes = (('text files', '*.txt'),
                 ('All files', '*.*'))

    file = asksaveasfilename(title='Save as a Text File',
                             initialdir='/',
                             filetypes=filetypes)

    text = t_in.get('1.0', END)

    with open(file + '.txt', 'w') as f:
        f.write(text)


def update_spkr(spkr_dict, sub_spkr_menu_list, spkr_dir):

    global spkr_name, spkr_img, spkr_author, spkr_web
    global spkr_readme, spkr_ver, init_spkr, spkr_lang
    global curr_spkr, curr_sub_spkr

    spkr_name.set(spkr_dict['speaker_name'])
    spkr_img.set(os.path.join(spkr_dir.get(), spkr_dict['img']))
    spkr_author.set(fxy(lang['Author'] + ': ' + spkr_dict['author']))
    spkr_web.set(spkr_dict['web'])
    spkr_readme.set(os.path.join(spkr_dir.get(), spkr_dict['readme']))
    spkr_ver.set(fxy(lang['Version'] + ': ' + spkr_dict['ver']))
    init_spkr.set(spkr_name.get())
    spkr_lang.set(spkr_dict['lang'])
    curr_spkr.set(spkr_name.get())
    curr_sub_spkr.set(sub_spkr_menu_list[0])


class neuTalk():

    def __init__(self):
        super().__init__()

        global lang_list, lang, default_dict
        global spkr_dir, pause_length, pho_dict
        global sub_spkr_dir_list, sub_spkr_menu_list
        global spkr_dir, spkr_name, spkr_img, spkr_author, spkr_web
        global spkr_readme, spkr_ver, init_spkr, spkr_lang, lang_menu
        global curr_spkr, curr_sub_spkr, normalize
        global sub_spkr_menu_list, sub_spkr_dir_list

        #root = ttkB.Window(themename='ntk_c')
        root = ttkB.Window(themename='ntk_c')

        spkr_dir = StringVar()
        spkr_name = StringVar()
        spkr_img = StringVar()
        spkr_author = StringVar()
        spkr_web = StringVar()
        spkr_ver = StringVar()
        spkr_readme = StringVar()
        init_spkr = StringVar()
        curr_spkr = StringVar()
        curr_sub_spkr = StringVar()
        spkr_lang = StringVar()
        pause_length = IntVar()
        pho_dict = StringVar()
        curr_sub_spkr = StringVar()
        normalize = StringVar()

        # globalparam.get_defaults()

        spkr_dir, pause_length, normalize, default_dict = get_defaults(
            spkr_dir, pause_length, pho_dict, normalize)
        lang_menu, lang_list, lang = lang_init(default_dict)
        spkr_dict, sub_spkr_dir_list, sub_spkr_menu_list = load_spkr(
            spkr_dir.get(), 0)
        update_spkr(spkr_dict, sub_spkr_menu_list, spkr_dir)

        main_window(root)

    def quit_pro(self):

        self.destroy()

    def about_win():

        global neutalk_ver

        abt_win = ttkB.Toplevel()
        self.abt_win.geometry(center(abt_win, 300, 200))
        self.abt_win.title('About')
        self.abt_win.focus_force()
        self.abt_win.grab_set()

        self.table_frame = ttkB.Frame(abt_win)
        self.table_frame.pack(padx=5, pady=5, fill=X, side=BOTTOM)

        data = [
            ('NeMo', '1.8.0'),
            ('pyworld', '0.3.0'),
            ('ControllableTalkNet', '1.2'),
            ('hifi-gan', 'N/A')
        ]

        tv = ttkB.Treeview(
            master=table_frame, columns=[0, 1], show=HEADINGS, height=5
        )
        for row in data:
            self.tv.insert('', END, values=row)

        self.tv.selection_set('I001')
        self.tv.heading(0, text='Dependency')
        self.tv.heading(1, text='Version')
        self.tv.column(0, width=200)
        self.tv.column(1, width=100, anchor=CENTER)
        self.tv.pack(side=LEFT, anchor=NE, fill=X)

        text = ('neuTalk (c) neutrogic 2022' + '\n' +
                'Current Version: ' + neutalk_ver)


def settings_win():

    global init_spkr
    global curr_lang
    global sprk_name
    global spkr_menu_list
    global lang
    global lang_list
    global default_dict
    global normalize

    curr_lang = StringVar()
    curr_lang.set(default_dict['lang'])

    set_win = ttkB.Toplevel()
    set_win.geometry(center(set_win, 400, 250))
    set_win.title(fxy(lang['Settings']))
    set_win.iconbitmap('img/orange.ico')
    set_win.focus_force()
    set_win.grab_set()

    # speaker frame
    speaker_frame = ttkB.Labelframe(
        master=set_win, width=390, height=240, text=fxy(lang['Speaker'])
    )
    speaker_frame.pack(side=TOP, pady=5, padx=5, fill=X, expand=True)

    # Default Speaker Selection
    defl_spkr_name = ttkB.Label(
        master=speaker_frame, text=fxy(lang['Default Speaker'] + ':')
    )
    defl_spkr_name.pack(side=LEFT, padx=5, pady=5)

    defl_spkr_sel = ttkB.Combobox(
        master=speaker_frame, width=33, textvariable=init_spkr, state='readonly', value=spkr_menu_list
    )
    defl_spkr_sel.current(spkr_menu_list.index(init_spkr.get()))
    defl_spkr_sel.pack(side=RIGHT, padx=5, pady=5)
    defl_spkr_sel.bind('<<ComboboxSelected>>', change_default_speaker)

    # general frame
    gen_set = ttkB.Labelframe(
        master=set_win, width=390, height=240, text=fxy(lang['Settings'])
    )
    gen_set.pack(side=TOP, pady=5, padx=5, fill=X, expand=True)

    # GUI Language Selector
    gui_lang_name = ttkB.Label(
        master=gen_set, text=fxy(lang['Display Language'] + ':')
    )
    gui_lang_name.pack(side=LEFT, padx=5, pady=5)

    gui_lang_sel = ttkB.Combobox(
        master=gen_set, width=33, textvariable=curr_lang, state='readonly', value=lang_list
    )
    gui_lang_sel.current(lang_list.index(curr_lang.get()))
    gui_lang_sel.pack(side=RIGHT, padx=5, pady=5)
    gui_lang_sel.bind('<<ComboboxSelected>>',
                      lambda: change_default_lang(set_win))

    audio_frame = ttkB.Labelframe(
        master=set_win, width=390, height=240, text=fxy(lang['Audio'])
    )
    audio_frame.pack(side=TOP, pady=5, padx=5, fill=X, expand=True)

    normalize_check = ttkB.Checkbutton(
        bootstyle='round-toggle', master=audio_frame, onvalue='yes',
        offvalue='no', text=fxy(lang['Normalize Synthesized Audio']), variable=normalize
    )
    normalize_check.pack(side=LEFT, padx=5, pady=5)


def change_default_speaker(event):

    global default_dict
    global default_path
    global spkr_menu_list
    global spkr_dir_list
    global init_spkr

    curr = spkr_dir_list[spkr_menu_list.index(init_spkr.get())]

    name = curr.split('speakers')[1]

    default_dict['speaker'] = 'speakers' + name

    try:
        with open(default_path, 'w') as f:
            for key in default_dict.keys():
                f.write('%s,%s\n' % (key, default_dict[key]))
    except IOError:
        print('I/O Error')

    print('[DEBUG] Updated Default Params')

    del curr
    del name
    gc.collect()


def main_window(root):

    root.geometry(center(root, 1, 1))  # Centers window with function
    root.iconbitmap('img/orange.ico')

    global spkr_icon
    global spkr_lbl_frame
    global spkr_menu_list
    global spkr_dir_list
    global asset_dir
    global sub_spkr_dir_list
    global sub_spkr_menu_list

    global pro_dir
    global asset_dir
    global curr_csv

    global spkr_dir
    global spkr_name
    global spkr_img
    global spkr_author
    global spkr_web
    global spkr_models
    global spkr_ver
    global spkr_readme
    global pause_length
    global syn_pitch
    global syn_speed
    global syn_volume
    global syn_breath
    global init_spkr
    global spkr_dict
    global curr_spkr
    global curr_sub_spkr
    global pho_dict
    global lang
    global lang_list, lang_menu
    global transf_var
    global spkr_lang
    global default_dict
    global normalize

    global t_in

    syn_pitch = IntVar()
    syn_volume = IntVar()
    syn_speed = IntVar()
    syn_breath = IntVar()
    transf_var = IntVar()

    syn_pitch.set(10)
    syn_speed.set(10)
    syn_volume.set(5)
    syn_breath.set(0)
    transf_var.set(0)

    # Sets up menu.
    menu_bar = Menu(master=root, background='gray12', fg='white')
    root.config(menu=menu_bar)

    file_menu = Menu(menu_bar, tearoff=0)
    file_menu.add_command(label=fxy(lang['Export Text']), command=export_txt)
    # file_menu.add_command(label=fxy(lang['Exit']),command=quit_pro)
    menu_bar.add_cascade(label=fxy(lang['File']), menu=file_menu)

    sett_menu = Menu(menu_bar, tearoff=0)
    sett_menu.add_command(label=fxy(lang['Settings']), command=settings_win)
    #sett_menu.add_command(label=fxy(lang['Dictionary Editor']), command=self.dictionary_editor)
    #sett_menu.add_command(label=fxy(lang['Library Format Tool']), command=lambda: libtool(lang))
    menu_bar.add_cascade(label=fxy(lang['Settings']), menu=sett_menu)

    spkr_o_menu = Menu(menu_bar, tearoff=0)
    spkr_o_menu.add_command(
        label=fxy(lang['Install Speaker']), command=lambda: install_spkr(root, lang))
    menu_bar.add_cascade(label=fxy(lang['Speaker']), menu=spkr_o_menu)

    help_menu = Menu(menu_bar, tearoff=0)
    help_menu.add_command(label=fxy(lang['Help']), command=donashi)
    # help_menu.add_command(label=fxy(lang['About']),command=about_win)
    menu_bar.add_cascade(label=fxy(lang['Help']), menu=help_menu)

    ASSET = Path(__file__).parent / 'img'

    sta = StringVar()
    sta.set('Select a speaker.')

    # Variables for speaker/synthesis

    pho_dict = StringVar()

    # === Initialize Directories ===
    spkr_dir = StringVar()
    spkr_dir.set('speakers')  # Sets speaker directory
    # Sets the speaker directories in a list
    spkr_dir_list = speaker_list(spkr_dir.get())
    # Sets the speaker as the names in their csv files
    spkr_menu_list = menu_list(spkr_dir_list)
    # Asset directory, not necessary but idk maybe it will be?
    asset_dir = os.path.join(pro_dir, 'img')
    op_dir = os.path.join(pro_dir, 'o')  # Temp synthesis files directory

    #spkr_dir, pause_length, default_dict = get_defaults(spkr_dir, pause_length, pho_dict)

    root.title(fxy(lang['neuTalk']))

    # EVENT: Singer has changed
    def change_speaker(event):

        global iconic2
        global spkr_img
        global spkr_icon
        global spkr_lbl_frame

        spkr_dir.set(spkr_dir_list[spkr_menu_list.index(spkr_sel_combo.get())])
        spkr_dict, sub_spkr_dir_list, sub_spkr_menu_list = load_spkr(
            spkr_dir.get(), 0)
        update_spkr(spkr_dict, sub_spkr_menu_list, spkr_dir)

        if len(sub_spkr_menu_list) == 1:
            sub_model_box.configure(state='disabled')
        else:
            sub_model_box.configure(state='readonly', value=sub_spkr_menu_list)
            sub_model_box.current(
                sub_spkr_menu_list.index(curr_sub_spkr.get()))

        if spkr_lang.get() == 'eng':
            wav_trsf_check.configure(state=ACTIVE)
            wav_trsf_entry.configure(state=ACTIVE)
            wav_trsf_button.configure(state=ACTIVE)
        if spkr_lang.get() == 'jpn':
            wav_trsf_check.configure(state=DISABLED)
            wav_trsf_entry.configure(state=DISABLED)
            wav_trsf_button.configure(state=DISABLED)
        if spkr_lang.get() == 'zh':
            wav_trsf_check.configure(state=DISABLED)
            wav_trsf_entry.configure(state=DISABLED)
            wav_trsf_button.configure(state=DISABLED)
        if spkr_lang.get() == 'fr':
            wav_trsf_check.configure(state=DISABLED)
            wav_trsf_entry.configure(state=DISABLED)
            wav_trsf_button.configure(state=DISABLED)
        if spkr_lang.get() == 'eng_tt2':
            wav_trsf_check.configure(state=DISABLED)
            wav_trsf_entry.configure(state=DISABLED)
            wav_trsf_button.configure(state=DISABLED)
        if spkr_lang.get() == 'eng_tt2_arpa':
            wav_trsf_check.configure(state=DISABLED)
            wav_trsf_entry.configure(state=DISABLED)
            wav_trsf_button.configure(state=DISABLED)
        if spkr_lang.get() == 'pipeline':
            wav_trsf_check.configure(state=DISABLED)
            wav_trsf_entry.configure(state=DISABLED)
            wav_trsf_button.configure(state=DISABLED)

        icon1 = Image.open(os.path.join(spkr_dir.get(), spkr_dict['img'])).resize(
            (120, 120), Image.ANTIALIAS)
        iconic1 = ImageTk.PhotoImage(icon1)

        spkr_icon.forget()
        del spkr_icon
        spkr_icon = ttkB.Label(
            master=spkr_lbl_frame, image=iconic1
        )
        spkr_icon.image = iconic1
        spkr_icon.pack(side=RIGHT, pady=5)
        root.update_idletasks()

    def change_sub_model(event):

        # Translates the name back into the directory
        curr = sub_spkr_dir_list[sub_spkr_menu_list.index(sub_model_box.get())]
        spkr_dir.set(curr)

        # Load models for synthesis
        if spkr_lang.get() == 'eng':
            talknet.load_talknet_models(curr)

        elif spkr_lang.get() == 'jpn':
            talknet.load_tacotron_model(curr)

        elif spkr_lang.get() == 'zh':
            talknet.load_tacotron_model(curr)
        elif spkr_lang.get() == 'fr':
            talknet.load_tacotron_model(curr)
        elif spkr_lang.get() == 'eng_tt2':
            talknet.load_tacotron_model(curr)
        elif spkr_lang.get() == 'eng_tt2_arpa':
            talknet.load_tacotron_model(curr)
        elif spkr_lang.get() == 'pipeline':
            talknet.load_pipeline_model(curr)

    def synthesize(text):

        global pro_dir
        global pause_length
        global syn_pitch
        global syn_speed
        global spkr_lang
        global default_dict

        import stitcher as stitcher

        combi = os.path.join(pro_dir, 'o/tmp_combi.wav')
        resynth = os.path.join(pro_dir, 'o/tmp_combi-resynth.wav')

        if os.path.exists(combi):
            os.remove(combi)
        if os.path.exists(resynth):
            os.remove(resynth)

        if transf_var.get() == 1:
            print('[DEBUG] Wav Transfer selected')
            wav = transfer_wav_dir.get()
            audio = talknet.wav_transfer(text, 1, wav)
            write(os.path.join('o/tmp0.wav'), 22050, audio)
        else:
            pau = AudioSegment.silent(duration=pause_length.get())

            text_list = stitcher.text_cutter(text)

            for i in range(len(text_list)):
                # Infers with Hifigan as the priority
                if spkr_lang.get() == 'eng':
                    audio = talknet.synthesize(text_list[i])
                elif spkr_lang.get() == 'jpn':
                    audio = talknet.synthesize_jpn(text_list[i])
                elif spkr_lang.get() == 'zh':
                    audio = talknet.synthesize_zh(text_list[i])
                elif spkr_lang.get() == 'fr':
                    audio = talknet.synthesize_fr(text_list[i])
                elif spkr_lang.get() == 'eng_tt2':
                    audio = talknet.synthesize_eng_tt2(text_list[i])
                elif spkr_lang.get() == 'eng_tt2_arpa':
                    audio = talknet.synthesize_eng_tt2_arpa(text_list[i])
                elif spkr_lang.get() == 'pipeline':
                    audio = talknet.synthesize_pipeline(text_list[i])

                # writes audio depending on which model was used to synthesize
                write(os.path.join('o/tmp' + str(i) + '.wav'), 22050, audio)

        stitcher.audio_stitch(pause_length.get(),
                              syn_volume.get(), normalize.get())

        if transf_var.get() != 1:
            if syn_pitch.get() != 10 or syn_speed.get() != 10 or syn_breath.get() != 0:

                world_neutalk.world_initialize(combi)

                if syn_pitch.get() != 10:
                    world_neutalk.repitch(syn_pitch.get())
                if syn_speed.get() != 10:
                    world_neutalk.respeed(syn_speed.get())
                if syn_breath.get() != 0:
                    world_neutalk.breath(syn_breath.get())
                # world_neutalk.expression(1)

                world_neutalk.world_write(
                    syn_pitch.get(), syn_speed.get(), syn_breath.get())

    def play_audio():

        # synthesizes audio
        audio = synthesize(t_in.get('1.0', END))

        if syn_pitch.get() != 10 or syn_speed.get() != 10:
            wav_path = os.path.join(pro_dir, 'o/tmp_combi-resynth.wav')
        else:
            wav_path = os.path.join(pro_dir, 'o/tmp_combi.wav')

        file = sa.WaveObject.from_wave_file(wav_path)
        play = file.play()
        del file

    def save_audio():

        # synthesizes audio
        audio = synthesize(t_in.get('1.0', END))

        # Prompts file save location
        type = [('Wave', '*.wav'),
                ('All Files', '*.*')]
        savedir = asksaveasfilename(
            filetypes=type, defaultextension=type, confirmoverwrite=True)

        # Writes to a file depending on which model is active
        write(savedir, 22050, audio)

    def import_txt():

        # Prompts to select file
        filetypes = (('text files', '*.txt'),
                     ('All files', '*.*'))
        file = askopenfilename(title='Open a Text File',
                               initialdir='/',
                               filetypes=filetypes)

        # Reads text file
        text = ''
        with open(file, 'r+') as f:
            text = f.read()

        # Deletes text in the t_in box and inserts the file
        t_in.delete('1.0', END)
        t_in.insert('1.0', text)

    def stop_audio():
        # Stops any audio playing from neuTalk
        sa.stop_all()

    # I don't have this currently implemented? might add it back later
    def open_spkr_dir():
        os.system('explorer.exe ' + spkr_cfg.curr_spkr_dir)

    def open_site():
        webbrowser.open_new_tab(spkr_web.get())

    def open_readme():
        # os.system('open ' + shlex.quote(spkr_cfg.readme)) #MacOS/X
        os.system('start ' + spkr_readme.get())  # Windows

    def trsf_wav_set():

        # Sets the fileset of the wav to be used in the talknet wav transfer feature!
        filetypes = (('wav file', '*.wav'),
                     ('All files', '*.*'))
        file = askopenfilename(title='Open a wav file containing vocals.',
                               initialdir='/',
                               filetypes=filetypes)

        transfer_wav_dir.set(file)

    # ============= BEGIN GUI ============================================================================

    # MAIN FRAMES ===================================================================

    left_frame = ttkB.Frame(master=root)
    left_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

    right_frame = ttkB.Frame(master=root)
    right_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nw')

    # Text Frame ====================================================================
    # Text Entry (it's not what i want.... grumble grumble)
    text_frame = ttkB.Frame(
        master=left_frame, width=370, height=200
    )
    text_frame.pack(side=TOP, padx=5, pady=5)

    t_in = ttkB.Text(text_frame,
                     font=('Roboto', 9),
                     wrap=WORD,
                     bd=0)
    if spkr_lang.get() == 'eng':
        t_in.insert(
            '1.0', "In this text box, enter any text you'd like to hear and the selected speaker will {R IY D} it aloud.")
    if spkr_lang.get() == 'jpn':
        t_in.insert(
            '1.0', "このテキストの枠に、聞きたいテキストを入力すると選んだスピーカーが朗読します。")
    if spkr_lang.get() == 'zh':
        t_in.insert(
            '1.0', "在这个文本框中，输入你想听到的任何文本，所选的发言人将大声朗读它。")
    if spkr_lang.get() == 'fr':
        t_in.insert(
            '1.0', "Dans cette zone de texte, saisissez le texte que vous souhaitez entendre et l'orateur sélectionné le lira à haute voix.")
    if spkr_lang.get() == 'eng_tt2':
        t_in.insert(
            '1.0', "In this text box, enter any text you'd like to hear and the selected speaker will read it aloud.")
    if spkr_lang.get() == 'eng_tt2_arpa':
        t_in.insert(
            '1.0', "In this text box, enter any text you'd like to hear and the selected speaker will reed it aloud.")
    if spkr_lang.get() == 'pipeline':
        t_in.insert(
            '1.0', "In this text box, enter any text you'd like to hear and the selected speaker will {R IY1 D} it aloud.")
    t_in.place(x=0, y=0, height=200, width=370)
    # =================================================================================

    # Buttons ==========================================================================
    button_frame = ttkB.Labelframe(
        master=left_frame, width=370, height=60, text=fxy(lang['Controls'])
    )
    button_frame.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    play_button = ttkB.Button(
        master=button_frame, width=9, text=fxy(lang['Play']), command=play_audio
    )
    play_button.pack(side=LEFT, padx=5, pady=7, expand=True)

    stop_button = ttkB.Button(
        master=button_frame, width=9, text=fxy(lang['Stop']), command=stop_audio
    )
    stop_button.pack(side=LEFT, padx=5, pady=7, expand=True)

    save_button = ttkB.Button(
        master=button_frame, width=9, text=fxy(lang['Save .wav']), command=save_audio
    )
    save_button.pack(side=LEFT, padx=5, pady=7, expand=True)

    text_button = ttkB.Button(
        master=button_frame, width=9, text=fxy(lang['Open .txt']), command=import_txt
    )
    text_button.pack(side=LEFT, padx=5, pady=7, expand=True)
    # =====================================================================================

    # Voice Parameters Box ========================================================================
    voice_param_box = ttkB.Labelframe(
        master=left_frame, width=370, height=270, text=fxy(lang['Voice Parameters']), padding=5
    )
    voice_param_box.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    # Pitch Slider
    def pitch_val_int(event):
        value = pitch_slider.get()
        if int(value) != value:
            pitch_slider.set(round(value))

    # slider
    pitch_slider = ttkB.Scale(
        master=voice_param_box, length=145, from_=20, to=1, variable=syn_pitch, orient=VERTICAL,
        command=pitch_val_int
    )
    pitch_slider.grid(row=0, column=0, padx=5, pady=5, sticky='n')

    # entry box
    pitch_val_label = ttkB.Entry(
        master=voice_param_box, textvariable=str(syn_pitch), width=5
    )
    pitch_val_label.bind('<Key>', verify_int)
    pitch_val_label.grid(row=1, column=0, padx=5, pady=5, sticky='n')

    # label
    pitch_sl_label = ttkB.Label(
        master=voice_param_box, text=fxy(lang['Pitch'])
    )
    pitch_sl_label.grid(row=2, column=0, padx=5, pady=5, sticky='n')

    # Speed Slider

    def speed_val_int(event):
        value = speed_slider.get()
        if int(value) != value:
            speed_slider.set(round(value))

    # slider
    speed_slider = ttkB.Scale(
        master=voice_param_box, length=145, from_=20, to=1, variable=syn_speed, orient=VERTICAL,
        command=speed_val_int)
    speed_slider.grid(row=0, column=1, padx=5, pady=5, sticky='n')

    # entry box
    speed_val_label = ttkB.Entry(
        master=voice_param_box, textvariable=str(syn_speed), width=5
    )
    speed_val_label.bind('<Key>', verify_int)
    speed_val_label.grid(row=1, column=1, padx=5, pady=5, sticky='n')

    # label
    speed_sl_label = ttkB.Label(
        master=voice_param_box, text=fxy(lang['Speed'])
    )
    speed_sl_label.grid(row=2, column=1, padx=5, pady=5, sticky='n')

    # volume Slider
    def vol_val_int(event):
        value = vol_slider.get()
        if int(value) != value:
            vol_slider.set(round(value))

    # slider
    vol_slider = ttkB.Scale(
        master=voice_param_box, length=145, from_=10, to=1, variable=syn_volume,
        orient=VERTICAL, command=vol_val_int
    )
    vol_slider.grid(row=0, column=2, padx=5, pady=5, sticky='n')

    # entry box
    vol_val_label = ttkB.Entry(
        master=voice_param_box, textvariable=str(syn_volume), width=5,
    )
    vol_val_label.bind('<Key>', verify_int)
    vol_val_label.grid(row=1, column=2, padx=5, pady=5, sticky='n')

    # label
    vol_sl_label = ttkB.Label(
        master=voice_param_box, text=fxy(lang['Volume'])
    )
    vol_sl_label.grid(row=2, column=2, padx=5, pady=5, sticky='n')

    # Pause Length Slider
    def pau_val_int(event):
        value = pau_slider.get()
        if int(value) != value:
            pau_slider.set(round(value))

    # slider
    pau_slider = ttkB.Scale(
        master=voice_param_box, length=145, from_=300, to=1, variable=pause_length,
        orient=VERTICAL, command=pau_val_int
    )
    pau_slider.grid(row=0, column=3, padx=5, pady=5, sticky='n')

    # entry box
    pau_val_label = ttkB.Entry(
        master=voice_param_box, textvariable=pause_length, width=5
    )
    pau_val_label.bind('<Key>', verify_int)
    pau_val_label.grid(row=1, column=3, padx=5, pady=5, sticky='n')

    # label
    pau_sl_label = ttkB.Label(
        master=voice_param_box, text=fxy(lang['Pause'])
    )
    pau_sl_label.grid(row=2, column=3, padx=2, pady=5, sticky='n')

    # Breath Factor Slider
    def bre_val_int(event):
        value = bre_slider.get()
        if int(value) != value:
            bre_slider.set(round(value))

    bre_slider = ttkB.Scale(
        master=voice_param_box, length=145, from_=10, to=0, variable=syn_breath,
        orient=VERTICAL, command=bre_val_int
    )
    bre_slider.grid(row=0, column=4, padx=5, pady=5, sticky='n')

    bre_val_label = ttkB.Entry(
        master=voice_param_box, textvariable=syn_breath, width=5,
    )
    bre_val_label.bind('<Key>', verify_int)
    bre_val_label.grid(row=1, column=4, padx=5, pady=5, sticky='n')

    bre_sl_label = ttkB.Label(
        master=voice_param_box, text=fxy(lang['Breath'])
    )
    bre_sl_label.grid(row=2, column=4, padx=5, pady=5, sticky='n')

    # ===========================================================================================

    # Speaker Info Frame =================================================================
    # Define frame
    speaker_space = ttkB.Labelframe(
        master=right_frame, width=370, height=450, text=fxy(lang['Speaker'])
    )
    speaker_space.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    sml_spk_frm = ttkB.Frame(
        master=speaker_space, width=360
    )
    sml_spk_frm.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    #Speaker: Label
    spkr_sel_label = ttkB.Label(
        master=sml_spk_frm, text=fxy(lang['Speaker']) + ': '
    )
    spkr_sel_label.pack(side=LEFT, padx=5, pady=5)

    # Speaker Select ComboBox
    spkr_sel_combo = ttkB.Combobox(
        master=sml_spk_frm, textvariable=curr_spkr, state='readonly',
        value=spkr_menu_list, width=38
    )
    spkr_sel_combo.current(spkr_menu_list.index(spkr_name.get()))
    spkr_sel_combo.pack(side=RIGHT, padx=5, pady=5, expand=True)
    spkr_sel_combo.bind('<<ComboboxSelected>>', change_speaker)

    # Speaker Label Frame ==================================================
    spkr_lbl_frame = ttkB.Labelframe(
        master=speaker_space, width=360, height=130, text=fxy(lang['Current Speaker'])
    )
    spkr_lbl_frame.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    # Icon
    icon = Image.open(spkr_img.get()).resize((120, 120), Image.ANTIALIAS)
    iconic = ImageTk.PhotoImage(icon, master=spkr_lbl_frame)

    spkr_icon = ttkB.Label(
        master=spkr_lbl_frame, image=iconic
    )
    spkr_icon.pack(side=RIGHT, pady=5)

    spkr_left = ttkB.Frame(
        master=spkr_lbl_frame
    )
    spkr_left.pack(side=LEFT, padx=5, pady=5)

    info_frm = ttkB.Frame(
        master=spkr_left, width=230
    )
    info_frm.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    # Speaker Name
    spkr_name_lbl = ttkB.Label(
        master=info_frm, textvariable=spkr_name
    )
    spkr_name_lbl.pack(side=TOP, padx=5, pady=3)

    # Author Name
    author_lbl = ttkB.Label(
        master=info_frm, textvariable=spkr_author
    )
    author_lbl.pack(side=TOP, padx=5, pady=3)

    # Version Name
    version_lbl = ttkB.Label(
        master=info_frm, textvariable=spkr_ver
    )
    version_lbl.pack(side=TOP, padx=5, pady=3)

    # subframe for button
    buttn_frm = ttkB.Frame(
        master=spkr_left, width=230
    )
    buttn_frm.pack(side=TOP, padx=5, pady=5)

    # site button
    site_button = ttkB.Button(
        master=buttn_frm, text=fxy(lang['Site']), width=9, command=open_site
    )
    site_button.pack(side=LEFT, padx=5, pady=5, expand=True)

    # readme button
    readme_button = ttkB.Button(
        master=buttn_frm, text=fxy(lang['ReadMe']), width=9, command=open_readme
    )
    readme_button.pack(side=LEFT, padx=5, pady=5, expand=True)
    # ================================================================================

    # TalkNet settings frame ===================================================================
    # this can be changed when other synthesizers are added and have different settings

    talknet_frame = ttkB.Labelframe(
        master=speaker_space, width=280, height=229, text=fxy(lang['Model Controls'])
    )
    talknet_frame.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    sub_model_frm = ttkB.Frame(
        master=talknet_frame, width=270
    )
    sub_model_frm.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)

    # Sub model lbl
    sub_model_lbl = ttkB.Label(
        master=sub_model_frm, text=fxy(lang['Sub-Model'] + ': ')
    )
    sub_model_lbl.pack(side=LEFT, padx=5, pady=5)

    # sub-model combo box
    sub_model_box = ttkB.Combobox(
        master=sub_model_frm, width=28, textvariable=curr_sub_spkr, state='readonly',
        values=sub_spkr_menu_list
    )

    if len(sub_spkr_menu_list) == 1:
        sub_model_box.configure(state='disabled')
    else:
        sub_model_box.configure(state='readonly')

    sub_model_box.current(sub_spkr_menu_list.index(curr_sub_spkr.get()))

    sub_model_box.pack(side=RIGHT, padx=5, pady=5)
    sub_model_box.bind('<<ComboboxSelected>>', change_sub_model)

    wav_trsf_frame = ttkB.Labelframe(
        master=talknet_frame, text=fxy(lang['Wav Transfer Controls']), padding=5
    )
    wav_trsf_frame.pack(side=TOP, padx=5, pady=5, expand=True)

    # wav transfer check box
    wav_trsf_check = ttkB.Checkbutton(
        bootstyle='round-toggle', master=wav_trsf_frame, onvalue=1,
        offvalue=0, text=fxy(lang['Use Wav Transfer Function']), variable=transf_var
    )
    wav_trsf_check.grid(row=0, column=0, columnspan=2,
                        padx=5, pady=5, sticky='w')

    # Variable for the directory of the wav
    transfer_wav_dir = StringVar()

    # wav transfer dir entry box
    wav_trsf_entry = ttkB.Entry(
        master=wav_trsf_frame, text=fxy(lang['Select a .wav file']),
        textvariable=transfer_wav_dir, width=33
    )
    wav_trsf_entry.grid(row=1, column=0, padx=5, pady=5, sticky='w')

    # wav transfer browse box
    wav_trsf_button = ttkB.Button(
        master=wav_trsf_frame, text=fxy(lang['Browse']), width=9, command=trsf_wav_set
    )
    wav_trsf_button.grid(row=1, column=1, padx=5, pady=5, sticky='w')

    if spkr_lang.get() == 'eng':
        wav_trsf_check.configure(state=ACTIVE)
        wav_trsf_entry.configure(state=ACTIVE)
        wav_trsf_button.configure(state=ACTIVE)
    if spkr_lang.get() == 'jpn':
        wav_trsf_check.configure(state=DISABLED)
        wav_trsf_entry.configure(state=DISABLED)
        wav_trsf_button.configure(state=DISABLED)
    if spkr_lang.get() == 'zh':
        wav_trsf_check.configure(state=DISABLED)
        wav_trsf_entry.configure(state=DISABLED)
        wav_trsf_button.configure(state=DISABLED)
    if spkr_lang.get() == 'fr':
        wav_trsf_check.configure(state=DISABLED)
        wav_trsf_entry.configure(state=DISABLED)
        wav_trsf_button.configure(state=DISABLED)
    if spkr_lang.get() == 'eng_tt2':
        wav_trsf_check.configure(state=DISABLED)
        wav_trsf_entry.configure(state=DISABLED)
        wav_trsf_button.configure(state=DISABLED)
    if spkr_lang.get() == 'eng_tt2_arpa':
        wav_trsf_check.configure(state=DISABLED)
        wav_trsf_entry.configure(state=DISABLED)
        wav_trsf_button.configure(state=DISABLED)
    if spkr_lang.get() == 'pipeline':
        wav_trsf_check.configure(state=DISABLED)
        wav_trsf_entry.configure(state=DISABLED)
        wav_trsf_button.configure(state=DISABLED)
    # ===========================================================================================================

    logo1 = Image.open('img/lil_logo.png')
    logoic1 = ImageTk.PhotoImage(logo1)

    # neuTalk logo @ da bottom
    ntk_logo_lbl = ttkB.Label(
        master=right_frame, image=logoic1
    )
    ntk_logo_lbl.pack(side=TOP, padx=5, pady=5)

    informations = ttkB.Label(
        master=right_frame, text=u'neuTalk (C) neutrogic 2022 | version: v0.6a', justify=RIGHT,
        foreground='gray22'
    )
    informations.pack(side=TOP, padx=5, pady=2)

    root.geometry(center(root, 755, 575))  # Centers window with function
    root.resizable(False, False)
    root.mainloop()


def main():
    app = neuTalk()


if __name__ == '__main__':
    main()
