import socket
import sys

IPADRESS = socket.gethostname()
PORTNUM = 5555

def sendMove(move):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        s.sendto(move, (IPADRESS, PORTNUM))
    except socket.error:
        print(socket.error)
        print(f'could not send move {move} to ip {IPADRESS}:{PORTNUM}')
        pass

    s.close()