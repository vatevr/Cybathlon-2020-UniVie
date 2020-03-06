
class SlidingWindowMaker:
    def __init__(self, updateFreq=200, windowSize=3000, fs=500):
        """
        SlidingWindowMaker is a tool used to compose a sliding window for online processing of the BCI pipeline.
        :param updateFreq: update frequency of the window in miliseconds.
        :param windowSize: size of the window in miliseconds.
        :param fs: sampling frequency in Hertz.
        """
        self.fs=fs
        self.updateFreq = updateFreq
        self.windowSize = windowSize
        self.windowBuffer = []

    def update(self, sample):
        if self.check_full():
            tmpbuf = self.windowBuffer
            #print(len(self.windowBuffer))
            self.windowBuffer = self.windowBuffer[self.updateFreq:]
            #print(len(self.windowBuffer))
            self.windowBuffer.append(sample)
            return tmpbuf
        else:
            self.windowBuffer.append(sample)
            return None

    def check_full(self):
        if len(self.windowBuffer) >= self.windowSize:
            return True
        else:
            return False
