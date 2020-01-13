import numpy as np
import scipy.linalg
import scipy.io
import scipy.signal


class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		self.channelCount = 0
		self.samplingFrequency = 0
		self.epochSampleCount = 0
		self.signalBuffer = list()
		self.signalHeader = None

	def initialize(self):
		self.channelCount = int(self.setting['Channel count'])
		self.samplingFrequency = int(self.setting['Sampling frequency'])
		self.epochSampleCount = int(self.setting['Generated epoch sample count'])

		self.dft = scipy.linalg.dft(self.epochSampleCount)
		self.idft = np.linalg.inv(self.dft)
		self.hilbert = np.zeros(self.epochSampleCount)

		delta = list(self.setting['delta'].split("-"))
		theta = list(self.setting['theta'].split("-"))
		alpha = list(self.setting['alpha'].split("-"))
		beta = list(self.setting['beta'].split("-"))
		gamma = list(self.setting['gamma'].split("-"))
		self.bands = {'delta': delta, 'theta': theta, 'alpha': alpha, 'beta': beta, 'gamma': gamma}

		if self.epochSampleCount % 2 == 0:
		    self.hilbert[0] = self.hilbert[self.epochSampleCount // 2] = 1
		    self.hilbert[1:self.epochSampleCount // 2] = 2
		else:
		    self.hilbert[0] = 1
		    self.hilbert[1:(self.epochSampleCount + 1) // 2] = 2
		if self.channelCount > 1:
		    ind = [np.newaxis] * 2
		    ind[-1] = slice(None)
		    self.hilbert = self.hilbert[tuple(ind)]	
	
	def do(self, x):
		x = np.asarray(x).T
		if x.shape[0] != self.epochSampleCount and x.shape[1] != self.channelCount:
		    raise ValueError("configs (",self.epochSampleCount, ",", self.channelCount, ") do not match input dims ", x.shape)
		if np.iscomplexobj(x):
		    raise ValueError("x is not a real signal.")
		H = self.dft.dot(x) * self.hilbert.T
		instant_frequency = np.ones((len(self.bands), self.channelCount), dtype='float')
		for i, band in enumerate(self.bands):
		    signal = np.zeros((self.epochSampleCount, self.channelCount), dtype=complex)
		    from_val = int(round(float(self.epochSampleCount) / float(self.samplingFrequency) * int(self.bands[band][0])))
		    to_val = int(round(float(self.epochSampleCount) / float(self.samplingFrequency) * int(self.bands[band][1])))
		    # uncomment to check whather a band is too narrow for the precision
		    #if to_val-from_val == 0:
		    #	raise ValueError(band, from_val, to_val, self.bands[band][0],  self.bands[band][1])
		    signal[from_val:to_val, :] = H[from_val:to_val, :]
		    signal = signal.T.dot(self.idft).T
		    inst_phase = np.unwrap(np.angle(signal))
		    inst_freq = np.diff(inst_phase, axis=0) / (2 * np.pi) * self.samplingFrequency
		    instant_frequency[i,:] = np.median(inst_freq, axis=0)
		return instant_frequency
	
	def process(self):
		for chunkIndex in range( len(self.input[0]) ):
			if(type(self.input[0][chunkIndex]) == OVSignalHeader):
				self.signalHeader = self.input[0].pop()
				
				outputHeader = OVSignalHeader(
				self.signalHeader.startTime, 
				self.signalHeader.endTime, 
				[len(self.bands), self.channelCount], 
				self.bands.keys() + self.channelCount*[''],
				self.signalHeader.samplingRate)
				
				self.output[0].append(outputHeader)
				
			elif(type(self.input[0][chunkIndex]) == OVSignalBuffer):
				chunk = self.input[0].pop()
				numpyBuffer = np.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
				self.signalBuffer = self.do(numpyBuffer).flatten().tolist()
				chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, self.signalBuffer)
				self.output[0].append(chunk)
				
			elif(type(self.input[0][chunkIndex]) == OVSignalEnd):
				self.output[0].append(self.input[0].pop())	 			

box = MyOVBox()
