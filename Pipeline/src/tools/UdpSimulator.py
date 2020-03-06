
import socket
import time

import numpy as np

from src.tools.FileReader import FileReader


class UdpSimulator:
    def __init__(self, samples_per_block=4, fs=500):
        self.samples_per_block=samples_per_block
        self.fs=fs

    def simulate_udp(self, path, address, port):
        reader = FileReader(datapath=path)
        #data = reader.load_mat()
        data, y, meta = reader.load_moabb()
        data =np.transpose(data, (1, 0, 2))
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        addr = (address, port)
        #nchn = 32
        print(addr)
        #data = data[:nchn]
        print(data.shape)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        counter = 0
        try:
            while True:
                #start = time.time()
                header = np.asarray(([0, counter, 0, 0, 0, 0, 0])).ravel().astype('>i4').tobytes('C')                           #transforms data to 4 byte integer!

                mysample = header+data[:, counter:counter + 3].T.ravel().astype('>i4').tobytes('C')     #possible loss of data here!

                sock.sendto(mysample, addr)
                time.sleep(4. / self.fs)
                counter=counter+1
                #end = time.time()
                #print(end-start)
        finally:
            print("done")