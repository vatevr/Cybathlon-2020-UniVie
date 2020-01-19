import constants
from controller import Controller
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from watchdog.events import FileSystemEventHandler

class LogListener(FileSystemEventHandler):
    def __init__(self, player_tag, ai_turned_on, track_length=None, f_e_d=None, s_e_d=None, t_e_d=None):
        self.player_tag = player_tag
        self.ai_turned_on = ai_turned_on
        self.last_modified = datetime.now()
        self.general_logs = []
        self.player_logs = []

        if self.ai_turned_on == True:
            self.enemy_tags = constants.PLAYERS_TAGS
            self.enemy_tags.remove(self.player_tag)
            self.enemy_logs = {
                self.enemy_tags[0]: [],
                self.enemy_tags[1]: [],
                self.enemy_tags[2]: [],
            }
            self.enemy_controllers = {
                self.enemy_tags[0]: Controller(self.enemy_tags[0], track_length, f_e_d),
                self.enemy_tags[1]: Controller(self.enemy_tags[1], track_length, s_e_d),
                self.enemy_tags[2]: Controller(self.enemy_tags[2], track_length, t_e_d),
            }

    def __general_logs(self, line):
        if 'start race' in line and line not in self.general_logs:
            print(f'Line: {line}', end='')
            print('race started!', end='\n\n')
            self.general_logs.append(line)
        elif 'puase race' in line and line not in self.general_logs:
            print(f'Line: {line}', end='')
            print('race paused!', end='\n\n')
            self.general_logs.append(line)

    def __player_logs (self, line):
        if (self.player_tag + '_expectedInput') in line:
            if 'none because end of curve' not in line:
                if line not in self.player_logs:
                    self.player_logs.append(line)
                    print(f'Line: {line}', end='')
                    if 'leftWinker' in line:
                        print(f'{self.player_tag} needs to send move: left', end='\n\n')
                        pass
                    elif 'headlight' in line:
                        print(f'{self.player_tag} needs to send move: headlights', end='\n\n')
                        pass
                    elif 'rightWinker' in line:
                        print(f'{self.player_tag} needs to send move: right', end='\n\n')
                        pass

    def process_enemy_logs (self, enemy_tag, line):
        if (enemy_tag + '_expectedInput') in line:
            if 'none because end of curve' not in line:
                if line not in self.enemy_logs[enemy_tag]:
                    self.enemy_logs[enemy_tag].append(line)
                    print(f'Line: {line}', end='')
                    self.enemy_controllers[enemy_tag].make_move(line)
                    # with ProcessPoolExecutor() as executor:
                    #     executor.submit(self.enemy_controllers[enemy_tag].make_move, line)

    def on_modified(self, event):
        # if time between modifications is smaller than 1 second, skip
        if datetime.now() - self.last_modified < timedelta(seconds=1):
            return
        # if modified file is not the race log file, skip
        elif 'raceLog' not in  event.src_path:
            return
        else:
            self.last_modified = datetime.now()
        # print(f'Event type: {event.event_type}  path : {event.src_path}')
        log_file  = open(event.src_path, 'r')
        for line in log_file:
            self.__general_logs(line)
            self.__player_logs(line)
            if self.ai_turned_on == True:
                self.process_enemy_logs(self.enemy_tags[0], line)
                self.process_enemy_logs(self.enemy_tags[1], line)
                self.process_enemy_logs(self.enemy_tags[2], line)
