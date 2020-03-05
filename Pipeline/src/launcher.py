from src.connector.udp_connector import UdpConnector
from src.classifier.FBCSP.FBCSPclassifier import classifier


file = file.open()

def score():
    pass


def write_chunk(window):
    pass


def routine(window):
    write_chunk(window)
    score()


def complete_recording():
    file.save()
    classifier.classify(file)


def main():
    connector = UdpConnector(batch_received=lambda window: routine(window), on_complete=lambda: complete_recording())
    connector.start()


if __name__ == "__main__":
    main()