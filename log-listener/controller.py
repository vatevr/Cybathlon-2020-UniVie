import time
import random
import parallel
import constants
import numpy as np
from connection import sendMove

class Controller:
    def __init__(self, player_tag, dif_index, config_reader, folder_path, eeg_amp):
        self.player_tag = player_tag

        if player_tag == 'p1':
            self.moves = constants.P1_MOVES
            self.eeg = constants.P1_EEG
        elif player_tag == 'p2':
            self.moves = constants.P2_MOVES
            self.eeg = constants.P2_EEG
        elif player_tag == 'p3':
            self.moves = constants.P3_MOVES
            self.eeg = constants.P3_EEG
        else:
            self.moves = constants.P4_MOVES
            self.eeg = constants.P4_EEG

        self.difficulty = config_reader.get_enemy_difficulty(dif_index)
        self.track_length = config_reader.get_track_length()
        self.min_delay = config_reader.get_min_delay()
        self.max_delay = config_reader.get_max_delay()
        self.number_of_ones = int(self.track_length * self.difficulty)
        self.number_of_zeros = (self.track_length - self.number_of_ones)
        self.track = np.ones(self.track_length)
        self.track[:self.number_of_zeros] = 0
        np.random.shuffle(self.track)
        self.track = self.track.tolist()
        print(f'{self.player_tag}: {self.number_of_ones} / {self.track_length}')
        print(f'{self.track}')
        self.enemy_logs = []
        self.enemy_logs_file = open(f"{folder_path}/enemy_{self.player_tag}_log", "w")
        self.eeg_amp = eeg_amp

    def __del__(self):
        for line in self.enemy_logs:
            self.enemy_logs_file.write(line)
        self.enemy_logs_file.close()

    def __send_wrong_move(self, line):
        if 'none' in line:
            wrong_move = random.choice(['leftWinker', 'headlight', 'rightWinker'])
            sendMove(self.moves[wrong_move])
            self.enemy_logs.append(f'{self.player_tag} sent wrong move: {wrong_move} \n\n')
            print(f'{self.player_tag} sent wrong move: {wrong_move}', end='\n\n')

    def __send_right_move(self, line):
        if 'leftWinker' in line:
            self.eeg_amp.setData(self.eeg['leftWinker'])
            sendMove(self.moves['leftWinker'])
            self.enemy_logs.append(f'{self.player_tag} sent move: left \n\n')
            print(f'{self.player_tag} sent move: left', end='\n\n')
        elif 'headlight' in line:
            self.eeg_amp.setData(self.eeg['headlight'])
            sendMove(self.moves['headlight'])
            self.enemy_logs.append(f'{self.player_tag} sent move: headlight \n\n')
            print(f'{self.player_tag} sent move: headlight', end='\n\n')
        elif 'rightWinker' in line:
            self.eeg_amp.setData(self.eeg['rightWinker'])
            sendMove(self.moves['rightWinker'])
            self.enemy_logs.append(f'{self.player_tag} sent move: right \n\n')
            print(f'{self.player_tag} sent move: right', end='\n\n')
        else:
            self.eeg_amp.setData(self.eeg['none'])
            self.enemy_logs.append(f'{self.player_tag} did not send any move \n\n')
            print(f'{self.player_tag} did not send any move', end='\n\n')

    def make_move(self, line):
        self.enemy_logs.append(f'Line: {line}')
        print(f'Line: {line}')
        val = self.track.pop(0)
        print(f'{self.player_tag}_track length: {len(self.track)}/{self.track_length}')
        if val == 1.0:
            self.__send_right_move(line)
        elif val == 0.0:
            chance = random.randint(0, 100) / 100.0
            print(f'{self.player_tag} has a chance of {chance} <= {self.difficulty}')
            if chance <= self.difficulty:
                self.__send_wrong_move(line)
                delay = round(random.uniform(self.min_delay, self.max_delay), 1)
                self.enemy_logs.append(f"{self.player_tag} delaying {delay} seconds \n")
                print(f"{self.player_tag} delaying {delay} seconds \n")
                time.sleep(delay)
                self.__send_right_move(line)
            else:
                self.enemy_logs.append(f'{self.player_tag} did not send any move \n\n')
                print(f'{self.player_tag} did not send any move', end='\n\n')