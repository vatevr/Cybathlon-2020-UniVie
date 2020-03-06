import socket
import numpy as np

from queue import Queue
from threading import Thread
from time import sleep, time
from signalprocessing.peakfrequency.peak_frequency import PeakFrequency
from signalprocessing import signal_tools
from riemannianClassifier.classifier import riemannianClassifier

# load .env file
from os import getenv
from dotenv import load_dotenv
load_dotenv()

n_chn = int(getenv('N_CHN'))
n_sample_per_block = int(getenv('N_SAMPLE_PER_BLOCK'))
windowlength = int(getenv('WINDOWLENGTH'))            # window length in seconds
fs = int(getenv('FS'))                    # sampling frequency of NeurOne
peak_frequency_o = PeakFrequency(channels=n_chn - 1, samples=(windowlength * fs), fs=fs)
clf = riemannianClassifier()
clf.load_self() #assuming that we have a pre-trained classifier
    


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

    # signal processing
    peak_frequency_result = peak_frequency_o.transform(x=window)
    welch = signal_tools.extract_amplitudes_welch(window, (windowlength * fs))

    print('{} {}'.format(peak_frequency_result.shape, welch.shape))

    binary_probabilities = clf.predict_proba(window) #compute class probabilities for binary classification

    # TO-DO pass probabilities to feedback/UI
    # ...

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
    sock.bind((getenv('UDP_IP'), int(getenv('UDP_PORT'))))

    # sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
    # tcp_server_address = (TCP_IP, TCP_PORT)
    # sock_tcp.bind(tcp_server_address)
    # sock_tcp.listen(1)

    packet_len = 28

    data_queue = Queue(maxsize=0) #setting the queue maxsize to 0 makes it "infinite". PriorityQueue could be interesting to maintain ordering of UDP packets.
    trigger_queue = Queue(maxsize=0)


    receiver_thread = Thread(target=receive, args=(sock, data_queue, trigger_queue))
    consumer_thread = Thread(target=process_window, args=(data_queue, trigger_queue))

    receiver_thread.start()
    consumer_thread.start()

    receiver_thread.join()
    consumer_thread.join()

