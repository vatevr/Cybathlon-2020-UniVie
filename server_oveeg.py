import socket
import sys
import numpy as np
import time
from loaddata import loadeeg
import signalprocessing.peakfrequency.peak_frequency as sp
'''
# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
# Bind the socket to the port
server_address = ('192.168.200.240', 10016)

print('starting up on %s port %s' % server_address, sys.stderr)
sock.bind(server_address)
fs = 500
nchn = 16
'''

#data_buffer = np.empty((nchn, 0))
# Listen for incoming connections
#sock.listen(1)
#full_data = loadeeg()
full_data = loadsample()
#datamat = full_data[:nchn]
print(full_data.shape)
'''
if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim!")

len_off_data = datamat.shape[1]

while True:
    # Wait for a connection
    print('waiting for a connection', sys.stderr)
    connection, client_address = sock.accept()
    try:
        print ( 'connection from' + str(client_address), sys.stderr)
        counter = 0
        # Receive the data in small chunks and retransmit it
        while True:
            mysample = np.asarray(datamat[:nchn, counter:counter+4].T.ravel().astype('<f4').tobytes('C'))
	    
            print(datamat[:nchn, counter:counter+4].T.ravel()) 
            print(mysample.shape)
            pf = sp.PeakFrequency(nchn, 4, 500)
            pf.fit(np.transpose(mysample))
            connection.sendall(mysample)
            print('sending ', sys.stderr) # %datamat[:nchn, counter].ravel() # "" %mysample
            counter = 0 if counter == len_off_data - 4  else counter + 4 
            time.sleep(4. / fs)
    finally:
        # Clean up the connection
        connection.close()
      
'''
        
