import json
from constants import PLAYER_TAGS
from configparser import ConfigParser

class ConfigReader():
    def __init__(self):
        config_object = ConfigParser()
        config_object.read("config.ini")

        self.path = config_object.get('DEFAULT', 'game_server_path')
        self.player_tag = config_object.get('DEFAULT', 'player_tag')
        if self.player_tag not in PLAYER_TAGS:
            raise ValueError(f"player_tag can only be one of {PLAYER_TAGS}")
        self.eeg_amp_connected = config_object.getboolean('DEFAULT', 'eeg_amp_connected')
        self.first_opponent_on = config_object.getboolean('DEFAULT', 'first_opponent_on')
        self.first_opponent_difficulty = config_object.getfloat('DEFAULT', 'first_opponent_difficulty')
        self.second_opponent_on = config_object.getboolean('DEFAULT', 'second_opponent_on')
        self.second_opponent_difficulty = config_object.getfloat('DEFAULT', 'second_opponent_difficulty')
        self.third_opponent_on = config_object.getboolean('DEFAULT', 'third_opponent_on')
        self.third_opponent_difficulty = config_object.getfloat('DEFAULT', 'third_opponent_difficulty')
        self.min_delay = config_object.getfloat('DEFAULT', 'min_delay')
        self.max_delay = config_object.getfloat('DEFAULT', 'max_delay')

        print(f'game_server_path: {self.path}')
        print(f'player_tag: {self.player_tag}')
        print(f'eeg_amp_connected: {self.eeg_amp_connected}')
        print(f'first_opponent_on: {self.first_opponent_on}')
        print(f'first_opponent_difficulty: {self.first_opponent_difficulty}')
        print(f'second_opponent_on: {self.second_opponent_on}')
        print(f'second_opponent_difficulty: {self.second_opponent_difficulty}')
        print(f'third_opponent_on: {self.third_opponent_on}')
        print(f'third_opponent_difficulty: {self.third_opponent_difficulty}')
        print(f'min_delay: {self.min_delay}')
        print(f'max_delay: {self.max_delay}')

        track_file = open(self.path + '/trackData.json', 'r')
        track_obj = json.load(track_file)
        self.track_length = len(track_obj['segments'])
        print(f'track_length: {self.track_length}')

    def get_path(self):
        return self.path

    def get_player_tag(self):
        return self.player_tag

    
    def get_eeg_amp_connected(self):
        return self.eeg_amp_connected

    def get_opponent_on(self, index):
        if index == 'first':
            return self.first_opponent_on
        if index == 'second':
            return self.second_opponent_on
        if index == 'third':
            return self.third_opponent_on
        else:
            raise ValueError('index can only be one of ["first","second","third"]')

    def get_opponent_difficulty(self, index):
        if index == 'first':
            return self.first_opponent_difficulty
        if index == 'second':
            return self.second_opponent_difficulty
        if index == 'third':
            return self.third_opponent_difficulty
        else:
            raise ValueError('index can only be one of ["first","second","third"]')
    
    def get_min_delay(self):
        return self.min_delay
    
    def get_max_delay(self):
        return self.max_delay

    def get_track_length(self):
        return self.track_length