import os
import sys
import json
import time
import shutil
import constants
from log_listener import LogListener
from watchdog.observers import Observer

if __name__ == "__main__":
    print('Settings (Enter for default values)')
    print('Server path:', end=' ')
    path = input()
    if len(path) == 0:
        path = constants.DEFAULT_PATH
    print(f'Server path set to: "{constants.DEFAULT_PATH}"', end='\n\n')
    
    print("Player tag(one of ['p1', 'p2', 'p3', 'p4']):", end=' ')
    player_tag = input()
    if len(player_tag) == 0:
        player_tag = constants.DEFAULT_PLAYER_TAG
    elif player_tag not in constants.PLAYERS_TAGS:
        raise ValueError("Error: player_tag has to be one of: ['p1', 'p2', 'p3', 'p4']")
    print(f'Player tag set to: "{player_tag}"', end='\n\n')

    print('Should AI control enemy cars? (y/n):', end=' ')
    answer = input()
    if len(answer) == 0 or answer == 'y':
        ai_turned_on = True
        print('AI System set to True', end='\n\n')
    elif answer == 'n':
        ai_turned_on = False
        print('AI System set to False', end='\n\n')
    else:
        raise ValueError("Error: The answer has to be one of ['y', 'n']")

    if ai_turned_on == True:
        print('First enemy difficulty(Value between 0 and 1):', end=' ')
        f_e_d = input()
        if len(f_e_d) == 0:
            f_e_d = constants.DEFAULT_FIRST_ENEMY_DIFFICULTY
        elif float(f_e_d) < 0 or float(f_e_d) > 1:
            raise ValueError('Error: Difficulty must be between 0 and 1')
        else:
            f_e_d = float(f_e_d)
        print(f'First enemy difficulty set to: {f_e_d}', end='\n\n')

        print('Second enemy difficulty(Value between 0 and 1):', end=' ')
        s_e_d = input()
        if len(s_e_d) == 0:
            s_e_d = constants.DEFAULT_SECOND_ENEMY_DIFFICULTY
        elif float(s_e_d) < 0 or float(s_e_d) > 1:
            raise ValueError('Error: Difficulty must be between 0 and 1')
        else:
            s_e_d = float(s_e_d)
        print(f'Second enemy difficulty set to: {s_e_d}', end='\n\n')

        print('Third enemy difficulty(Value between 0 and 1):', end=' ')
        t_e_d = input()
        if len(t_e_d) == 0:
            t_e_d = constants.DEFAULT_THIRD_ENEMY_DIFFICULTY
        elif float(t_e_d) < 0 or float(t_e_d) > 1:
            raise ValueError('Error: Difficulty must be between 0 and 1')
        else:
            t_e_d = float(t_e_d)
        print(f'Third enemy difficulty set to: {t_e_d}', end='\n\n')
        
        track_file = open(path + '/trackData.json', 'r')
        track_obj = json.load(track_file)
        track_length = len(track_obj['segments'])

    if ai_turned_on == True:
        event_handler = LogListener(player_tag, ai_turned_on, track_length, f_e_d, s_e_d, t_e_d)
    else:
        event_handler = LogListener(player_tag, ai_turned_on)
    observer = Observer()
    observer.schedule(event_handler, path=(path + '/log'), recursive=False)
    observer.start()
    print()
    print('Now listening for log entries. Please start the race!', end='\n\n')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        shutil.rmtree(path + '/log')
        os.mkdir(path + '/log')
    observer.join()