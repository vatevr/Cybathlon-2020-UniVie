PLAYERS_TAGS = ['p1', 'p2', 'p3', 'p4']

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
    'none': 14
}

P2_EEG = {
    'leftWinker': 21,
    'headlight': 22,
    'rightWinker': 23,
    'none': 24
}

P3_EEG = {
    'leftWinker': 31,
    'headlight': 32,
    'rightWinker': 33,
    'none': 34
}

P4_EEG = {
    'leftWinker': 41,
    'headlight': 42,
    'rightWinker': 43,
    'none': 44
}

# PLAYER EEG DATA:
## LEFT:        11
## HEADLIGHTS:  12
## RIGHT:       13
## NONE:        14

# ENEMY 1 EEG DATA:
## LEFT:        21
## HEADLIGHTS:  22
## RIGHT:       23
## NONE:        24
## FINISHED:    25

# ENEMY 2 EEG DATA:
## LEFT:        31
## HEADLIGHTS:  32
## RIGHT:       33
## NONE:        34
## FINISHED:    35

# ENEMY 3 EEG DATA:
## LEFT:        41
## HEADLIGHTS:  42
## RIGHT:       43
## NONE:        44
## FINISHED:    45

# GAME RELATED EEG DATA:
## STARTED:     90
## PAUSED:      91
## UNPAUSED:    92