import socket
import sys
import numpy as np
import time
from loaddata import loadeeg


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
# Bind the socket to the port
server_address = ('192.168.200.240', 10005)

print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)
fs = 500
nchn = 16


data_buffer = np.empty((nchn, 0))
# Listen for incoming connections
sock.listen(1)
full_data = loadeeg()
datamat = full_data[:nchn]


if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim!")

len_off_data = datamat.shape[1]

while True:
    # Wait for a connection
    print >>sys.stderr, 'waiting for a connection'
    connection, client_address = sock.accept()
    try:
        print >>sys.stderr, 'connection from', client_address
        counter = 0
        # Receive the data in small chunks and retransmit it
        while True:
            mysample = datamat[:nchn, counter:counter+4].T.ravel().astype('<f4').tobytes('C')
            print datamat[:nchn, counter:counter+4].T.ravel()

            connection.sendall(mysample)
            print >>sys.stderr, 'sending ' # %datamat[:nchn, counter].ravel() # "" %mysample
            counter = 0 if counter == len_off_data - 4  else counter + 4 
            time.sleep(4. / fs)
    finally:
        # Clean up the connection
        connection.close()
      

        
