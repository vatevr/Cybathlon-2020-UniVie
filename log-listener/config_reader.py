import json
from configparser import ConfigParser

class ConfigReader():
    def __init__(self):
        config_object = ConfigParser()
        config_object.read("config.ini")

        self.path = config_object.get('DEFAULT', 'game_server_path')
        self.player_tag = config_object.get('DEFAULT', 'player_tag')
        if self.player_tag not in ['p1','p2','p3','p4']:
            raise ValueError("player_tag can only be one of ['p1','p2','p3','p4']")
        self.ai_turned_on = config_object.getboolean('DEFAULT', 'ai_turned_on')
        self.eeg_amp_connected = config_object.getboolean('DEFAULT', 'eeg_amp_connected')
        self.first_enemy_difficulty = config_object.getfloat('DEFAULT', 'first_enemy_difficulty')
        self.second_enemy_difficulty = config_object.getfloat('DEFAULT', 'second_enemy_difficulty')
        self.third_enemy_difficulty = config_object.getfloat('DEFAULT', 'third_enemy_difficulty')
        self.min_delay = config_object.getfloat('DEFAULT', 'min_delay')
        self.max_delay = config_object.getfloat('DEFAULT', 'max_delay')

        print(f'game_server_path: {self.path}')
        print(f'player_tag: {self.player_tag}')
        print(f'ai_turned_on: {self.ai_turned_on}')
        print(f'eeg_amp_connected: {self.eeg_amp_connected}')
        print(f'first_enemy_difficulty: {self.first_enemy_difficulty}')
        print(f'second_enemy_difficulty: {self.second_enemy_difficulty}')
        print(f'third_enemy_difficulty: {self.third_enemy_difficulty}')
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

    def get_ai_turned_on(self):
        return self.ai_turned_on
    
    def get_eeg_amp_connected(self):
        return self.eeg_amp_connected

    def get_enemy_difficulty(self, dif_index):
        if dif_index == 'first':
            return self.first_enemy_difficulty
        if dif_index == 'second':
            return self.second_enemy_difficulty
        if dif_index == 'third':
            return self.third_enemy_difficulty
        else:
            raise ValueError('dif_index can only be one of ["first","second","third"]')
    
    def get_min_delay(self):
        return self.min_delay
    
    def get_max_delay(self):
        return self.max_delay

    def get_track_length(self):
        return self.track_length