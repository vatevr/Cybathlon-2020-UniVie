import parallel

class EEG_AMP():
    def __init__(self, eeg_amp_connected):
        self.eeg_amp_connected = eeg_amp_connected
        if self.eeg_amp_connected:
            self.eeg_amp = parallel.Parallel()


    def setData(self, data):
        if self.eeg_amp_connected:
            self.eeg_amp.setData(data)
            print(f'Data: {data} is sent to EEG-AMP')