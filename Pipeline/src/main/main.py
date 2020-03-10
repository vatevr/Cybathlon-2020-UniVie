from Pipeline.src.connector.udp_connector import UdpConnector
from Pipeline.src.tools.testingUtility import printWindowCallback, printClassifierProbaCallback, sendToVisTest
from Pipeline.src.tools.UdpSimulator import UdpSimulator
from Pipeline.src.classifier.riemannianClassifier.classifier import riemannianClassifier
from Pipeline.src.tools.FileReader import FileReader
import socket
from threading import Thread


def offline_training(datapath, savepath) :
    reader = FileReader(datapath=datapath)
    X, y = reader.load_brainvision(classes=2)

    clf = riemannianClassifier(savePath=savepath)
    import pdb
    pdb.set_trace()
    clf.fit(X, y)


def online_training(path) :

    clf = riemannianClassifier(savePath=path)
    clf.load_self()

    simulator = UdpSimulator(fs=512)
    # simulator.simulate_udp("../../../../tcpip/data/s01.mat", address=IPAddr, port=10005)
    connector = UdpConnector(batch_received=sendToVisTest, n_chn=22, host="127.0.1.1", port=10005)
    # connector.start()

    broadcaster_thread = Thread(target=simulator.simulate_udp, args=(path, IPAddr, 10005))
    consumer_thread = Thread(target=connector.start, args=())

    broadcaster_thread.start()
    consumer_thread.start()

    broadcaster_thread.join()
    consumer_thread.join()

if __name__ == '__main__':
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    path = "../../../../tcpip/data/s01.mat"
    clfpath="../savedFilters/test"
    kspath="../../../../recordings/KS/2020_Cybathlon_KS_Session1.vhdr"

    offline_training(kspath, clfpath)
    print("classifier trained")
    online_training(clfpath)





