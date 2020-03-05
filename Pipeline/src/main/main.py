from Pipeline.src.connector.udp_connector import UdpConnector
from Pipeline.src.tools.testingUtility import printWindowCallback
from Pipeline.src.tools.UdpSimulator import UdpSimulator
import socket

if __name__ == '__main__':
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)

    simulator = UdpSimulator(fs=512)
    simulator.simulate_udp("../../../../tcpip/data/s01.mat", address=IPAddr, port=10005)
    connector = UdpConnector(batch_received=printWindowCallback)
    connector.start()


