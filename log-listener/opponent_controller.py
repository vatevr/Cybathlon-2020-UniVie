import constants
from opponent import Opponent

class OpponentController():
    def __init__(self, dependencies, folder_path, eeg_amp):
        self.opponent_logs = {}
        self.opponent_tags = []
        self.opponents = {}
        if dependencies.get_config_reader().get_opponent_on('first') == True:
            opponent_tag = dependencies.get_opponent_tags()[0]
            self.opponent_tags.append(opponent_tag)
            self.opponent_logs.update({opponent_tag: []})
            self.opponents.update({opponent_tag: Opponent(opponent_tag, 'first', dependencies.get_config_reader(), folder_path, eeg_amp)})
        if dependencies.get_config_reader().get_opponent_on('second') == True:
            opponent_tag = dependencies.get_opponent_tags()[1]
            self.opponent_tags.append(opponent_tag)
            self.opponent_logs.update({opponent_tag: []})
            self.opponents.update({opponent_tag: Opponent(opponent_tag, 'second', dependencies.get_config_reader(), folder_path, eeg_amp)})
        if dependencies.get_config_reader().get_opponent_on('third') == True:
            opponent_tag = dependencies.get_opponent_tags()[2]
            self.opponent_tags.append(opponent_tag)
            self.opponent_logs.update({opponent_tag: []})
            self.opponents.update({opponent_tag: Opponent(opponent_tag, 'third', dependencies.get_config_reader(), folder_path, eeg_amp)})

    def process_opponent_logs (self, line):
        for opponent_tag in self.opponent_tags:
                if (opponent_tag + '_expectedInput') in line:
                    if 'none because end of curve' not in line:
                        if line not in self.opponent_logs[opponent_tag]:
                            self.opponent_logs[opponent_tag].append(line)
                            self.opponents[opponent_tag].move(line)
                    break