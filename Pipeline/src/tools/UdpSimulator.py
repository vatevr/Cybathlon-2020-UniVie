from Pipeline.src.tools.FileReader import FileReader
import socket
import sys
import time
import numpy as np

class UdpSimulator:
    def __init__(self, samples_per_block=4, fs=500):
        self.samples_per_block=samples_per_block
        self.fs=fs

    def simulate_udp(self, path, address, port):
        reader = FileReader(datapath=path)
        data = reader.load_mat()

        addr = (address, port)
        nchn = 32
        data = data[:nchn]

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        counter = 1
        try:
            while True:
                mysample = data[:nchn, counter:counter + 4].T.ravel().astype('<f4').tobytes('C')
                #print(data[:nchn, counter:counter + 4].T.ravel())
                sock.sendto(mysample, addr)
                time.sleep(4. / self.fs)
        finally:
            print("done")