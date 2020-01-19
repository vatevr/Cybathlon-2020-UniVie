import os
import sys
import json
import time
import shutil
import constants
from log_listener import LogListener
from configparser import ConfigParser
from watchdog.observers import Observer

if __name__ == "__main__":
    config_object = ConfigParser()
    config_object.read("config.ini")

    path = config_object.get('DEFAULT', 'game_server_path')
    player_tag = config_object.get('DEFAULT', 'player_tag')
    ai_turned_on = config_object.getboolean('DEFAULT', 'ai_turned_on')
    f_e_d = config_object.getfloat('DEFAULT', 'first_enemy_difficulty')
    s_e_d = config_object.getfloat('DEFAULT', 'second_enemy_difficulty')
    t_e_d = config_object.getfloat('DEFAULT', 'third_enemy_difficulty')

    print(f'game_server_path: {path}')
    print(f'player_tag: {player_tag}')
    print(f'ai_turned_on: {ai_turned_on}')
    print(f'first_enemy_difficulty: {f_e_d}')
    print(f'second_enemy_difficulty: {s_e_d}')
    print(f'third_enemy_difficulty: {t_e_d}')

    if ai_turned_on == True:
        track_file = open(path + '/trackData.json', 'r')
        track_obj = json.load(track_file)
        track_length = len(track_obj['segments'])
        print(f'track_length: {track_length}')
        event_handler = LogListener(player_tag, ai_turned_on, track_length, f_e_d, s_e_d, t_e_d)
    else:
        event_handler = LogListener(player_tag, ai_turned_on)
    observer = Observer()
    observer.schedule(event_handler, path=(path + '/log'), recursive=False)
    observer.start()
    print('\nNow listening for log entries. Please start the race!', end='\n\n')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        shutil.rmtree(path + '/log')
        os.mkdir(path + '/log')
    observer.join()