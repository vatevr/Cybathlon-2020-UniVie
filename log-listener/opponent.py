import time
import random
import parallel
import constants
import numpy as np
from connection import sendMove

class Opponent():
    def __init__(self, player_tag, dif_index, config_reader, folder_path, eeg_amp):
        self.player_tag = player_tag
        self.move_set = constants.get_move_set_by_tag(self.player_tag)
        self.eeg_set = constants.get_eeg_set_by_tag(self.player_tag)

        self.difficulty = config_reader.get_opponent_difficulty(dif_index)
        self.track_length = config_reader.get_track_length()
        self.min_delay = config_reader.get_min_delay()
        self.max_delay = config_reader.get_max_delay()
        self.number_of_ones = int(self.track_length * self.difficulty)
        self.number_of_zeros = (self.track_length - self.number_of_ones)
        self.track = np.ones(self.track_length)
        self.track[:self.number_of_zeros] = 0
        np.random.shuffle(self.track)
        self.track = self.track.tolist()
        self.logs = []
        self.log_file = open(f"{folder_path}/opponent_{self.player_tag}_log", "w")
        self.eeg_amp = eeg_amp
        log = f'{self.player_tag}: {self.number_of_ones} / {self.track_length} \n'
        self.__print_and_add_log(log)
        log = f'{self.track} \n'
        self.__print_and_add_log(log)

    def __del__(self):
        for line in self.logs:
            self.log_file.write(line)
        self.log_file.close()

    def __print_and_add_log(self, log):
        self.logs.append(log)
        print(log)

    def move(self, line):
        line = line.replace("\n", "")
        log = f'Line: {line}\n'
        self.__print_and_add_log(log)
        move = None
        log = None
        if 'leftWinker' in line:
            move = 'leftWinker'
            log = f'{self.player_tag} needs to send move: {move}\n'
        elif 'headlight' in line:
            move = 'headlight'
            log = f'{self.player_tag} needs to send move: {move}\n'
        elif 'rightWinker' in line:
            move = 'rightWinker'
            log = f'{self.player_tag} needs to send move: {move}\n'
        elif 'none' in line:
            move = 'none'
            log = f'{self.player_tag} does not need to send any move: {move}\n'
        self.eeg_amp.setData(self.eeg_set[move])
        self.__print_and_add_log(log)
        self.__decide_move_behaviour(move)

    def __send_wrong_move_if_none(self, move):
        if move == 'none':
            wrong_move = random.choice(['leftWinker', 'headlight', 'rightWinker'])
            sendMove(self.move_set[wrong_move])
            self.eeg_amp.setData(self.eeg_set[wrong_move + 'Sent'])
            log = f'{self.player_tag} sent wrong move: {wrong_move} \n'
            self.__print_and_add_log(log)

    def __send_right_move(self, move):
        if move != 'none':
            sendMove(self.move_set[move])
        self.eeg_amp.setData(self.eeg_set[move + 'Sent'])
        log = None
        if move == 'none':
            log = f'{self.player_tag} did not send any move \n\n'
        else:
            log = f'{self.player_tag} sent move: {move} \n\n'
        self.__print_and_add_log(log)

    def __decide_move_behaviour(self, move):
        val = self.track.pop(0)
        if val == 1.0:
            self.__send_right_move(move)
        elif val == 0.0:
            chance = random.randint(0, 100) / 100.0
            log = f'{self.player_tag} has a chance of {chance} <= {self.difficulty} \n'
            self.__print_and_add_log(log)
            if chance <= self.difficulty:
                self.__send_wrong_move_if_none(move)
                delay = round(random.uniform(self.min_delay, self.max_delay), 1)
                log = f'{self.player_tag} delaying {delay} seconds \n'
                self.__print_and_add_log(log)
                time.sleep(delay)
                self.__send_right_move(move)
            else:
                log = f'{self.player_tag} could not send any move \n\n'
                self.__print_and_add_log(log)