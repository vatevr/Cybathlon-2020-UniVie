import os
import shutil
import constants
from eeg_amp import EEG_AMP
from datetime import datetime
from opponent_controller import OpponentController
from watchdog.events import FileSystemEventHandler

class LogListener(FileSystemEventHandler):
    def __init__(self, dependencies):
        self.player_tag = dependencies.get_config_reader().get_player_tag()
        self.last_modified = datetime.now()
        self.general_logs = []
        self.player_logs = []
        current_time = datetime.now()
        self.folder_path = f'sessions/{current_time}'
        os.makedirs(self.folder_path)
        self.general_logs_file = open(f"{self.folder_path}/general_logs", "w")
        self.player_logs_file = open(f"{self.folder_path}/player_logs", "w")
        self.src_path = ""
        self.eeg_amp = dependencies.get_eeg_amp()
        self.opponent_controller = OpponentController(dependencies, self.folder_path, self.eeg_amp)
        self.opponent_tags = dependencies.get_opponent_tags()

    def __del__(self):
        for line in self.general_logs:
            self.general_logs_file.write(line)
        self.general_logs_file.close()
        for line in self.player_logs:
            self.player_logs_file.write(line)
        self.player_logs_file.close()
        if self.src_path is not "":
            shutil.copy(self.src_path, self.folder_path)

    def __general_logs(self, line):
        if 'start race' in line and line not in self.general_logs:
            self.eeg_amp.setData(constants.RACE_EEG['started'])
            print(f'Line: {line}', end='')
            print('race started!', end='\n\n')
            self.general_logs.append(line)
        elif 'pause race' in line and line not in self.general_logs:
            self.eeg_amp.setData(constants.RACE_EEG['paused'])
            print(f'Line: {line}', end='')
            print('race paused!', end='\n\n')
            self.general_logs.append(line)
        elif 'resume paused race' in line and line not in self.general_logs:
            self.eeg_amp.setData(constants.RACE_EEG['unpaused'])
            print(f'Line: {line}', end='')
            print('race unpaused!', end='\n\n')
            self.general_logs.append(line)
        elif 'finish' in line and line not in self.general_logs:
            if self.player_tag in line:
                self.eeg_amp.setData(constants.get_eeg_by_tag(self.player_tag)['finished'])
            elif self.opponent_tags[0] in line:
                self.eeg_amp.setData(constants.get_eeg_by_tag(self.opponent_tags[0])['finished'])
            elif self.opponent_tags[1] in line:
                self.eeg_amp.setData(constants.get_eeg_by_tag(self.opponent_tags[1])['finished'])
            elif self.opponent_tags[2] in line:
                self.eeg_amp.setData(constants.get_eeg_by_tag(self.opponent_tags[2])['finished'])
            print(line)
            self.general_logs.append(line)

    def __player_logs (self, line):
        if (self.player_tag + '_expectedInput') in line:
            if 'none because end of curve' not in line:
                if line not in self.player_logs:
                    self.player_logs.append(line)
                    print(f'Line: {line}', end='')
                    player_eeg = constants.get_eeg_by_tag(self.player_tag)
                    if 'leftWinker' in line:
                        print(f'{self.player_tag} needs to send move: left', end='\n\n')
                        self.eeg_amp.setData(player_eeg['leftWinker'])
                        pass
                    elif 'headlight' in line:
                        print(f'{self.player_tag} needs to send move: headlights', end='\n\n')
                        self.eeg_amp.setData(player_eeg['headlight'])
                        pass
                    elif 'rightWinker' in line:
                        print(f'{self.player_tag} needs to send move: right', end='\n\n')
                        self.eeg_amp.setData(player_eeg['rightWinker'])
                        pass
                    elif 'none' in line:
                        self.eeg_amp.setData(player_eeg['none'])
                        pass

    def on_modified(self, event):
        # if modified file is not the race log file, skip
        if 'raceLog' not in  event.src_path:
            return
        else:
            if len(self.src_path) == 0:
                self.src_path = event.src_path
            log_file  = open(event.src_path, 'r')
            for line in log_file:
                self.__general_logs(line)
                self.__player_logs(line)
                self.opponent_controller.process_opponent_logs(line)