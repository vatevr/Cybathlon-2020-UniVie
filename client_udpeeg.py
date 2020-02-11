import socket
import numpy as np

from queue import Queue
from threading import Thread
from time import sleep, time

UDP_IP = "192.168.200.240"
UDP_PORT = 10015  # NeurOne

n_chn = 127
n_sample_per_block = 4
windowlength = 1            # window length in seconds
fs = 500                    # sampling frequency of NeurOne


def receive(sock, data_queue, trigger_queue):

    np_samples = np.zeros([n_sample_per_block, n_chn])
    prev_seq_nr = None
    while True:
        # connection, client_address = sock_tcp.accept()
        try:
            while True:

                cnt = 0
                while cnt < n_sample_per_block:

                    data, addr = sock.recvfrom(2048)  # buffer size is 1024 bytes
                    np_samples[cnt], label, sequence_nr = convert_bytes(data, cnt)
                    if prev_seq_nr is None:
                        prev_seq_nr = sequence_nr
                    elif sequence_nr != prev_seq_nr+1:
                        print("package out of order!")

                    prev_seq_nr = sequence_nr
                    cnt += 1
                    data_queue.put_nowait(np_samples)
                    trigger_queue.put_nowait(label)

                
                    #print(np_samples)
                    # print(len(np_samples))



        except():
            print("error receiving")
            exit()
        # finally:
            # connection.close
            # pass

def process_window(data_queue, trigger_queue):
    cnt = 0
    window = np.zeros([windowlength * fs, n_chn-1])
    while True:
        while cnt < windowlength*fs:
            start = time()
            block = data_queue.get(block=True)
            print(data_queue.qsize())
            for sample in range(block.shape[0]):
                window[cnt, :] = block[sample, :-1]
                cnt += 1
        do_stuff(window.T)
        stop = time()
        print(stop - start)
        cnt = 0



def do_stuff(window) :
    #print("ready to process window of length {} and {} ".format(window.shape, window))
    sleep(0.22)
    return

def convert_bytes(data, cnt):
    np_samples = np.zeros([n_sample_per_block, n_chn])
    samples = data[28:]
    np_samples[cnt] = np.asarray(
        [int.from_bytes(samples[x:x + 3], byteorder='big', signed=True) for x in range(0, len(samples), 3)])

    label = np_samples[cnt][-1]

    sequence_nr = int.from_bytes(data[4:8], byteorder='big', signed=True)

    return np_samples[cnt], label, sequence_nr



if __name__ == '__main__':

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet
    sock.bind((UDP_IP, UDP_PORT))

    # sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
    # tcp_server_address = (TCP_IP, TCP_PORT)
    # sock_tcp.bind(tcp_server_address)
    # sock_tcp.listen(1)

    packet_len = 28

    data_queue = Queue(maxsize=0) #setting the queue maxsize to 0 makes it "infinite". PriorityQueue could be interesting to maintain ordering of UDP packets.
    trigger_queue = Queue(maxsize=0)

    #receive(sock, data_queue, trigger_queue)

    receiver_thread = Thread(target=receive, args=(sock, data_queue, trigger_queue))
    consumer_thread = Thread(target=process_window, args=(data_queue, trigger_queue))

    receiver_thread.start()
    consumer_thread.start()

    receiver_thread.join()
    consumer_thread.join()

