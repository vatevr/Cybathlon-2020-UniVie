import socket

from queue import Queue
from threading import Thread
from time import sleep, time
import numpy as np


class UdpConnector:
    def __init__(self, batch_received, port=10015, host="192.168.200.240"):
        self.port = port
        self.host = host

        # callback executed on each windows received
        self.on_batch_received = batch_received
        
        # 
        self.n_chn = 127
        self.n_sample_per_block = 4
        self.windowlength = 1  # window length in seconds
        self.fs = 500  # sampling frequency of NeurOne

    def receive(self, sock, data_queue, trigger_queue):

        np_samples = np.zeros([self.n_sample_per_block, self.n_chn])
        prev_seq_nr = None
        while True:
            try:
                while True:

                    cnt = 0
                    while cnt < self.n_sample_per_block:

                        data, addr = sock.recvfrom(2048)  # buffer size is 1024 bytes
                        np_samples[cnt], label, sequence_nr = self.convert_bytes(data, cnt)
                        if prev_seq_nr is None:
                            prev_seq_nr = sequence_nr
                        elif sequence_nr != prev_seq_nr + 1:
                            print("package out of order!")

                        prev_seq_nr = sequence_nr
                        cnt += 1
                        data_queue.put_nowait(np_samples)
                        trigger_queue.put_nowait(label)

                        # print(np_samples)
                        # print(len(np_samples))

            except():
                print("error receiving")
                exit()


    def process_window(self, data_queue, trigger_queue):
        cnt = 0
        window = np.zeros([self.windowlength * self.fs, self.n_chn - 1])
        while True:
            while cnt < self.windowlength * self.fs:
                start = time()
                block = data_queue.get(block=True)
                for sample in range(block.shape[0]):
                    window[cnt, :] = block[sample, :-1]
                    cnt += 1
            
            self.on_batch_received(window.T)
            
            stop = time()
            print(stop - start)
            cnt = 0


    def convert_bytes(self, data, cnt):
        np_samples = np.zeros([self.n_sample_per_block, self.n_chn])
        samples = data[28:]
        np_samples[cnt] = np.asarray(
            [int.from_bytes(samples[x:x + 3], byteorder='big', signed=True) for x in range(0, len(samples), 3)])

        label = np_samples[cnt][-1]

        sequence_nr = int.from_bytes(data[4:8], byteorder='big', signed=True)

        return np_samples[cnt], label, sequence_nr

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet
        sock.bind((self.host, self.port))

        # sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
        # tcp_server_address = (TCP_IP, TCP_PORT)
        # sock_tcp.bind(tcp_server_address)
        # sock_tcp.listen(1)

        packet_len = 28
        
        # setting the queue maxsize to 0 makes it "infinite". PriorityQueue could be interesting to maintain ordering of UDP packets.
        data_queue = Queue(maxsize=0)  
        trigger_queue = Queue(maxsize=0)

        # receive(sock, data_queue, trigger_queue)

        receiver_thread = Thread(target=self.receive, args=(sock, data_queue, trigger_queue))
        consumer_thread = Thread(target=self.process_window, args=(data_queue, trigger_queue))

        receiver_thread.start()
        consumer_thread.start()

        receiver_thread.join()
        consumer_thread.join()
