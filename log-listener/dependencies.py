from constants import PLAYER_TAGS
from eeg_amp import EEG_AMP
from config_reader import ConfigReader

class Dependencies():
    def __init__(self):
        self.config_reader = ConfigReader()
        self.opponent_tags = PLAYER_TAGS
        self.opponent_tags.remove(self.config_reader.get_player_tag())
        self.eeg_amp = EEG_AMP(self.config_reader.get_eeg_amp_connected())
    
    def get_config_reader(self):
        return self.config_reader

    def get_opponent_tags(self):
        return self.opponent_tags

    def get_eeg_amp(self):
        return self.eeg_amp