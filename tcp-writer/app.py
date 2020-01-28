import socket

from pylsl import StreamInlet, resolve_stream
import requests


if __name__ == '__main__':
    """
    accepts all incoming streams over tcp 
    """
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    # inlet.pull_chunk()

    host, port = '127.0.0.1', 65432
    buffer_size = 1024

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))

        sock.listen()
        conn, addr = sock.accept()
        with conn:
            print(f'connected from {addr}')
            while True:
                data = conn.recv(buffer_size)
                if not data:
                    break
                conn.sendall(data)

        file = open('../eeg/recording.eeg', 'wb')
        file.write(data)

        '''
        writes files to server
        '''
        response = requests.post('http://localhost/api/record', files={'file': file})
