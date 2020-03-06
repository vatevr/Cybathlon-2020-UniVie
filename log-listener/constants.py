PLAYER_TAGS = ['p1', 'p2', 'p3', 'p4']

P1_MOVES = {
    'leftWinker': b'\x0B',
    'headlight': b'\x0C',
    'rightWinker': b'\x0D'
}

P2_MOVES = {
    'leftWinker': b'\x15',
    'headlight': b'\x16',
    'rightWinker': b'\x17'
}

P3_MOVES = {
    'leftWinker': b'\x1F',
    'headlight': b'\x20',
    'rightWinker': b'\x21'
}

P4_MOVES = {
    'leftWinker': b'\x29',
    'headlight': b'\x2A',
    'rightWinker': b'\x2B'
}

P1_EEG = {
    'leftWinker': 11,
    'headlight': 12,
    'rightWinker': 13,
    'none': 14,
    'finished': 15,
    'leftWinkerSent': 16,
    'headlightSent': 17,
    'rightWinkerSent': 18,
    'noneSent': 19, 
}

P2_EEG = {
    'leftWinker': 21,
    'headlight': 22,
    'rightWinker': 23,
    'none': 24,
    'finished': 25,
    'leftWinkerSent': 26,
    'headlightSent': 27,
    'rightWinkerSent': 28,
    'noneSent': 29,
}

P3_EEG = {
    'leftWinker': 31,
    'headlight': 32,
    'rightWinker': 33,
    'none': 34,
    'finished': 35,
    'leftWinkerSent': 36,
    'headlightSent': 37,
    'rightWinkerSent': 38,
    'noneSent': 39,
}

P4_EEG = {
    'leftWinker': 41,
    'headlight': 42,
    'rightWinker': 43,
    'none': 44,
    'finished': 45,
    'leftWinkerSent': 46,
    'headlightSent': 47,
    'rightWinkerSent': 48
    ,'noneSent': 49,
}

RACE_EEG = {
    'started': 90,
    'paused': 91,
    'unpaused': 92
}

def get_move_set_by_tag(tag):
    if tag == 'p1':
        return P1_MOVES
    elif tag == 'p2':
        return P2_MOVES
    elif tag == 'p3':
        return P3_MOVES
    else:
        return P4_MOVES

def get_eeg_set_by_tag(tag):
    if tag == 'p1':
        return P1_EEG
    elif tag == 'p2':
        return P2_EEG
    elif tag == 'p3':
        return P3_EEG
    else:
        return P4_EEG