import sys
import os
import csv
from pathlib import Path

#Classes for global parameters. they were making the main file cluttered.

global default_dict
default_dict = {}

global default_path
default_path = (os.path.join(Path.cwd(), 'default_param.csv'))

def get_defaults():
    
    global default_dict
    global default_path
   
    with open(default_path, 'r+') as f:
        reader = csv.reader(f)
        default_dict = {rows[0]:rows[1] for rows in reader}
    
    speaker_cfg.curr_spkr_dir = default_dict['speaker']
    syn_param.pause_length = int(default_dict['pause_length'])

class speaker_cfg():

    def __init__(self):
    
        global default_dict
    
        self._curr_spkr_dir = ''
        self._curr_spkr_csv = {}
        self._spkr_name = ''
        self._img = ''
        self._author = ''
        self._web = ''
        self._tt_model = ''
        self._hfg_model = ''
        self._ver = ''
        self._readme = ''
    
    @property
    def curr_spkr_dir(self):
        #print('[DEBUG] Called spkr_dir getter')
        return self._spkr_dir
    
    @curr_spkr_dir.setter
    def curr_spkr_dir(self, a):
        #print('[DEBUG] Called spkr_dir setter')
        self._spkr_dir = a
    
    @property
    def curr_spkr_csv(self):
        return self._curr_spkr_csv
    
    @curr_spkr_csv.setter
    def curr_spkr_csv(self, a):
        #print('[DEBUG] Called curr_spkr_csv setter')
        self._curr_spkr_csv = a
        
    @property
    def spkr_name(self):
        #print('[DEBUG] Called spkr_name get')
        return self._spkr_name
    
    @spkr_name.setter
    def spkr_name(self, a):
        #print('[DEBUG] Called spkr_name setter')
        self._spkr_name = a
    
    @property
    def img(self):
        #print('[DEBUG] Called img getter')
        return self._img
    
    @img.setter
    def img(self, a):
        #print('[DEBUG] Called img setter')
        self._img = a
    
    @property
    def author(self):
        #print('[DEBUG] Called author getter')
        return self._author
    
    @author.setter
    def author(self, a):
        #print('[DEBUG] Called author setter')
        self._author = a
    
    @property
    def web(self):
        #print('[DEBUG] Called web getter')
        return self._web
    
    @web.setter
    def web(self, a):
        #print('[DEBUG] Called web setter')
        self._web = a

    @property
    def tt_model(self):
        #print('[DEBUG] Called hfg_model getter')
        return self._tt_model
    
    @tt_model.setter
    def tt_model(self, a):
        #print('[DEBUG] Called hfg_model setter')
        self._tt_model = a
        
    @property
    def hfg_model(self):
        #print('[DEBUG] Called hfg_model getter')
        return self._hfg_model
    
    @hfg_model.setter
    def hfg_model(self, a):
        #print('[DEBUG] Called hfg_model setter')
        self._hfg_model = a

    @property
    def ver(self):
        #print('[DEBUG] Called ver getter')
        return self._ver
    
    @ver.setter
    def ver(self, a):
        #print('[DEBUG] Called ver setter')
        self._ver = a
        
    @property
    def readme(self):
        #print('[DEBUG] Called readme getter')
        return self._readme
    
    @readme.setter
    def readme(self, a):
        #print('[DEBUG] Called readme setter')
        self._readme = a

def syn_param():

    def __init__(self):
        
        global default_dict
        
        self._sigma = 0.666
        self._denoise = 0.01
        self._pause_length = 150
        self._pitch = 50
        self._speed = 50
        self._volume = 50
    
    @property
    def sigma(self):
        return self._sigma
        
    @sigma.setter
    def sigma(self, a):
        return self._sigma
        
    @property
    def denoise(self):
        return self._denoise
        
    @denoise.setter
    def denoise(self, a):
        return self._denoise
    
    @property
    def pause_length(self):
        return self._pause_length
        
    @pause_length.setter
    def pause_length(self, a):
        return self._pause_length
    
    @property
    def pitch(self):
        return self._pitch
        
    @pitch.setter
    def pitch(self, a):
        return self._pitch
    
    @property
    def speed(self):
        return self._speed
        
    @speed.setter
    def speed(self, a):
        return self._speed
        
    @property
    def volume(self):
        return self._volume
        
    @speed.setter
    def volume(self, a):
        return self._volume