
import socket
import numpy as np


import socket
import numpy as np

UDP_IP = "192.168.200.240"
UDP_PORT = 10023   #NeurOne 

TCP_IP = "192.168.200.240"
TCP_PORT = 10025  # Openvibe

n_chn = 126
n_sample_per_block = 512


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)# Internet
sock.bind((UDP_IP, UDP_PORT))

sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
tcp_server_address = (TCP_IP, TCP_PORT)
sock_tcp.bind(tcp_server_address)
sock_tcp.listen(1)


np_samples = np.zeros([n_sample_per_block, n_chn])

packet_len = 28

while True:
	connection, client_address = sock_tcp.accept()  
	try:
		while True:
			cnt = 0
			while  cnt < n_sample_per_block:
				data, addr = sock.recvfrom(2048) # buffer size is 1024 bytes
				samples = data[28:]
				np_samples[cnt] = np.asarray([int.from_bytes(samples[x:x+3], byteorder='big', signed=True) for x in range(0, len(samples),3 )])
				cnt += 1
			mysample = np_samples.ravel().astype('<i4').tobytes('C')
			print(np_samples)
			print(len(np_samples))
			connection.sendall(mysample)
			
	finally:
		connection.close
		pass
