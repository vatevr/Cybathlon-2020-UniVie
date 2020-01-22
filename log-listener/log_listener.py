import os
import shutil
import constants
from controller import Controller
from eeg_amp import EEG_AMP
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from watchdog.events import FileSystemEventHandler

class LogListener(FileSystemEventHandler):
    def __init__(self, player_tag, ai_turned_on, eeg_amp_connected, track_length=None, f_e_d=None, s_e_d=None, t_e_d=None, min_delay=1.0, max_delay=2.5):
        self.player_tag = player_tag
        self.ai_turned_on = ai_turned_on
        self.last_modified = datetime.now()
        self.general_logs = []
        self.player_logs = []
        current_time = datetime.now()
        self.folder_path = f'sessions/{current_time}'
        os.makedirs(self.folder_path)
        self.general_logs_file = open(f"{self.folder_path}/general_logs", "w")
        self.player_logs_file = open(f"{self.folder_path}/player_logs", "w")
        self.src_path = ""
        self.eeg_amp = EEG_AMP(eeg_amp_connected)

        if self.ai_turned_on == True:
            self.enemy_tags = constants.PLAYERS_TAGS
            self.enemy_tags.remove(self.player_tag)
            self.enemy_logs = {
                self.enemy_tags[0]: [],
                self.enemy_tags[1]: [],
                self.enemy_tags[2]: [],
            }
            self.enemy_controllers = {
                self.enemy_tags[0]: Controller(self.enemy_tags[0], track_length, f_e_d, min_delay, max_delay, self.folder_path, self.eeg_amp),
                self.enemy_tags[1]: Controller(self.enemy_tags[1], track_length, s_e_d, min_delay, max_delay, self.folder_path, self.eeg_amp),
                self.enemy_tags[2]: Controller(self.enemy_tags[2], track_length, t_e_d, min_delay, max_delay, self.folder_path, self.eeg_amp),
            }

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
            self.eeg_amp.setData(90)
            print(f'Line: {line}', end='')
            print('race started!', end='\n\n')
            self.general_logs.append(line)
        elif 'pause race' in line and line not in self.general_logs:
            self.eeg_amp.setData(91)
            print(f'Line: {line}', end='')
            print('race paused!', end='\n\n')
            self.general_logs.append(line)
        elif 'resume paused race' in line and line not in self.general_logs:
            self.eeg_amp.setData(92)
            print(f'Line: {line}', end='')
            print('race unpaused!', end='\n\n')
            self.general_logs.append(line)
        elif 'finish' in line and line not in self.general_logs:
            if self.player_tag in line:
                self.eeg_amp.setData(15)
            elif self.enemy_tags[0] in line:
                self.eeg_amp.setData(25)
            elif self.enemy_tags[1] in line:
                self.eeg_amp.setData(35)
            elif self.enemy_tags[2] in line:
                self.eeg_amp.setData(45)
            print(line)
            self.general_logs.append(line)

    def __player_logs (self, line):
        if (self.player_tag + '_expectedInput') in line:
            if 'none because end of curve' not in line:
                if line not in self.player_logs:
                    self.player_logs.append(line)
                    print(f'Line: {line}', end='')
                    if 'leftWinker' in line:
                        print(f'{self.player_tag} needs to send move: left', end='\n\n')
                        self.eeg_amp.setData(11)
                        pass
                    elif 'headlight' in line:
                        print(f'{self.player_tag} needs to send move: headlights', end='\n\n')
                        self.eeg_amp.setData(12)
                        pass
                    elif 'rightWinker' in line:
                        print(f'{self.player_tag} needs to send move: right', end='\n\n')
                        self.eeg_amp.setData(13)
                        pass
                    elif 'none' in line:
                        self.eeg_amp.setData(14)
                        pass

    def process_enemy_logs (self, enemy_tag, line):
        if (enemy_tag + '_expectedInput') in line:
            if 'none because end of curve' not in line:
                if line not in self.enemy_logs[enemy_tag]:
                    self.enemy_logs[enemy_tag].append(line)
                    self.enemy_controllers[enemy_tag].make_move(line)

    def on_modified(self, event):
        # if time between modifications is smaller than 1 second, skip
        # if datetime.now() - self.last_modified < timedelta(seconds=1):
        #     return

        # if modified file is not the race log file, skip
        if 'raceLog' not in  event.src_path:
            return
        else:
            self.last_modified = datetime.now()
            if len(self.src_path) == 0:
                self.src_path = event.src_path
            log_file  = open(event.src_path, 'r')
            for line in log_file:
                self.__general_logs(line)
                self.__player_logs(line)
                if self.ai_turned_on == True:
                    self.process_enemy_logs(self.enemy_tags[0], line)
                    self.process_enemy_logs(self.enemy_tags[1], line)
                    self.process_enemy_logs(self.enemy_tags[2], line)
