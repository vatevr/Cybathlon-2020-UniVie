import constants
from opponent import Opponent

class OpponentController():
    def __init__(self, dependencies, folder_path, eeg_amp):
        self.opponent_tags = dependencies.get_opponent_tags()
        self.opponent_logs = {
            self.opponent_tags[0]: [],
            self.opponent_tags[1]: [],
            self.opponent_tags[2]: [],
        }
        self.opponents = {
            self.opponent_tags[0]: Opponent(self.opponent_tags[0], 'first', dependencies.get_config_reader(), folder_path, eeg_amp),
            self.opponent_tags[1]: Opponent(self.opponent_tags[1], 'second', dependencies.get_config_reader(), folder_path, eeg_amp),
            self.opponent_tags[2]: Opponent(self.opponent_tags[2], 'third', dependencies.get_config_reader(), folder_path, eeg_amp),
        }

    def process_opponent_logs (self, line):
        for opponent_tag in self.opponent_tags:
                if (opponent_tag + '_expectedInput') in line:
                    if 'none because end of curve' not in line:
                        if line not in self.opponent_logs[opponent_tag]:
                            self.opponent_logs[opponent_tag].append(line)
                            self.opponents[opponent_tag].move(line)
                    break