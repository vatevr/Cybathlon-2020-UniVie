import time
import random
import constants
import numpy as np
from connection import sendMove

class Controller:
    def __init__(self, player_tag, track_length, difficulty):
        self.player_tag = player_tag

        if player_tag == 'p1':
            self.moves = constants.P1_MOVES
        elif player_tag == 'p2':
            self.moves = constants.P2_MOVES
        elif player_tag == 'p3':
            self.moves = constants.P3_MOVES
        elif player_tag == 'p4':
            self.moves = constants.P4_MOVES

        self.track_length = track_length
        self.difficulty = float(difficulty)
        self.number_of_ones = int(self.track_length * difficulty)
        self.number_of_zeros = (self.track_length - self.number_of_ones)
        self.track = np.ones(self.track_length)
        self.track[:self.number_of_zeros] = 0
        np.random.shuffle(self.track)
        self.track = self.track.tolist()
        print(f'{self.player_tag}: {self.number_of_ones} / {self.track_length}')
        print(f'{self.track}')

    def __send_right_move(self, line):
        if 'leftWinker' in line:
            sendMove(self.moves['leftWinker'])
            print(f'{self.player_tag} sent move: left', end='\n\n')
        elif 'headlight' in line:
            sendMove(self.moves['headlight'])
            print(f'{self.player_tag} sent move: headlight', end='\n\n')
        elif 'rightWinker' in line:
            sendMove(self.moves['rightWinker'])
            print(f'{self.player_tag} sent move: right', end='\n\n')

    def make_move(self, line):
        val = self.track.pop(0)
        print(f'{self.player_tag}_track length: {len(self.track)}/{self.track_length}')
        if len(self.track) == 0:
            print(f'{self.player_tag}: Track ended!')
        if val == 1.0:
            self.__send_right_move(line)
        elif val == 0.0:
            print(f'{self.player_tag} did not send any move')
            chance = random.randint(0, 100) / 100.0
            print(f'{self.player_tag} has a chance of {chance} <= {self.difficulty}')
            if chance <= self.difficulty:
                delay = random.randint(10, 25) / 10.0
                print(f"{self.player_tag} delaying {delay}")
                time.sleep(delay)
                self.__send_right_move(line)
            else:
                print()