import mne
import matplotlib.pyplot as plt
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
from mne.time_frequency import psd_multitaper
import numpy as np

def aggregate_events(events, skiplegs=False, sub=False):
	"""
	skiplegs - in case TF needs to ignore the legs trigger
	sub - in case TF is used, update the 5 minute trigger states
	"""
	temp_events = events[:]
	once = True # Because TF had both long rests in one phase and signal
	for i in range(len(events)):
		#print(events[0][i])
		if events[i][2] < 8: # Motor
			if skiplegs and events[i][2] == 6:
				continue
			temp_events[i][2] = 1
		elif events[i][2] < 30: # Non-Motor
			temp_events[i][2] = 2
		elif sub and not once:
			temp_events[i][2] = 31
			once = False
		elif sub and once:
			temp_events[i][2] = 32
	return temp_events
	
# Load data
def load_TF():
	raw1, picks, events1 = load_TF_2("data/TF/20191201_Cybathlon_TF_Session1_Block1.vhdr")
	raw2, picks, events2 = load_TF_2("data/TF/20191201_Cybathlon_TF_Session1_Block2.vhdr")
	raw3, _, events3 = load_TF_2("data/TF/20191201_Cybathlon_TF_Session1_RS.vhdr", sub=True)

	#events1 = aggregate_events(events1, True)
	#events2 = aggregate_events(events2)
	# Epoch data (cut up data into trials)
	tmin = 2			# time in seconds after trigger the trial should start
	tmax = tmin + 5	 # time in seconds after trigger the trial should end
	epochs1 = mne.Epochs(raw1, events1, tmin=tmin, tmax=tmax, preload=True, baseline=None, picks=picks)
	epochs2 = mne.Epochs(raw2, events2, tmin=tmin, tmax=tmax, preload=True, baseline=None, picks=picks)
	epochs3 = mne.Epochs(raw3, events3, tmin=tmin, tmax=tmax, preload=True, baseline=None, picks=picks)

	epochs = mne.concatenate_epochs([epochs1,epochs2,epochs3])
	return epochs, raw1
def load_TF_2(fname, sub=False):
	raw = mne.io.read_raw_brainvision(fname, preload=True)
	# Set montage (location of channels)
	raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
	montage = mne.channels.read_montage("standard_1005")
	raw.set_montage(montage)
	raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
	# Remove bad channels from analysis
	raw.info['bads'] = ['F2', 'FFC2h', 'TP8']#,'TPP8h']#, 'PPO6h']
	picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
	raw.set_eeg_reference(ref_channels="average")
	# Create events from triggers
	events = mne.events_from_annotations(raw)[0]
	events = aggregate_events(events, sub=sub)
	return raw, picks, events
def load_CA():
	raw = mne.io.read_raw_brainvision("data/CA/20191210_Cybathlon_CA_Session1.vhdr", preload=True)
	# Set montage (location of channels)
	raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
	montage = mne.channels.read_montage("standard_1005")
	raw.set_montage(montage)
	raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
	# Remove bad channels from analysis
	raw.info['bads'] = ["PPO9h", "FFT7h"]
	picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
	raw.set_eeg_reference(ref_channels="average")
	# Create events from triggers
	events = mne.events_from_annotations(raw)[0]
	events = aggregate_events(events) # Aggregate
	tmin = 2			# time in seconds after trigger the trial should start
	tmax = tmin + 5	 # time in seconds after trigger the trial should end
	epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True, baseline=None, picks=picks)
	return epochs, raw
	
epochs,raw = load_CA()
#epochs,raw = load_TF()

#mne.viz.plot_raw_psd(raw) # Do if you have lots of free memory ;)
#plt.show()
"""
for epoch in epochs.event_id:
	if epoch == "99999":
		continue
	print(epoch)
	mne.viz.plot_raw_psd(epochs[epoch].average())
	plt.show()
"""

bands1 =  [(1, 4, 'Delta (1 - 4)')]
bands2 =  [(4, 7, 'Theta (4 - 7)')]
bands3 =  [(8, 12, 'Alpha (8 - 12)')]
bands4 =  [(53, 60, 'Gamma (53 - 60)')]
#epochs["6"].plot_psd_topomap(bands=bands3)


psd_rest, freqs = psd_multitaper(epochs["30"], fmax=200)
ch_type = mne.channels._get_ch_type(epochs, None)
picks, pos, merge_grads, names, ch_type = mne.viz.topomap._prepare_topo_plot(epochs, ch_type, None)

# plotting per task  type
for trig in epochs.event_id:
	if trig == "40" or trig == "99999": continue
	print(trig)
	
	# Plot psd for the different task types
	psd_trig, _ = psd_multitaper(epochs[trig], fmax=200)
	psd_diff = psd_trig.mean(0) - psd_rest.mean(0)
	"""
	plt.plot(psd_trig.mean(0)) # Average psds over all epochs
	plt.show()
	
	plt.plot(psd_diff) # Average psds over all epochs
	plt.show()
	"""
	# Plot topographies in selected frequency bands for selected conditions
	
	fig = mne.viz.topomap.plot_psds_topomap(psd_trig.mean(0), freqs=freqs, pos=pos, show=False)
	fig.set_size_inches((40,10))
	fig.savefig("topo_"+str(trig)+".png")
	
	fig = mne.viz.topomap.plot_psds_topomap(psd_diff + np.ones((psd_diff.shape[0],1001)), freqs=freqs, pos=pos, show=False)
	# 124,1001 for CA, 123,1001 for TF due to bad channel removal
	# ones are added because the plotting logarithms, which mean 0 differences are problematic
	
	fig.set_size_inches((40,10))
	fig.savefig("topo_diff_"+str(trig)+".png")
	#mne.viz.plot_topomap(psd_trig.mean(0)-psd_rest.mean(0), pos=pos)
	plt.show()
	

# psd per electrode per task type
for pick in ["C3","C4","Pz","Fz","F3"]:
	for trig in epochs.event_id:
		if trig == "1":
			c = "b"
		elif trig == "2":
			c = "g"
		elif trig == "30":
			c = "r"
		else:
			continue
		axes = plt.axes()
		fig = epochs[trig].plot_psd(fmax=70.0, ax = axes, show=False, picks = [pick], color=c, area_alpha=0.5)
		fig.set_label(trig)

	axes.figure.set_size_inches((20,20))
	axes.figure.savefig(str(pick)+".png")
	print(pick)
	plt.show()
	

# Overall average psd
for trig in epochs.event_id:
	if trig == "1":
		c = "b"
	elif trig == "2":
		c = "g"
	elif trig == "30":
		c = "r"
	else:
		continue
	axes = plt.axes()
	fig = epochs[trig].plot_psd(fmax=70.0, ax = axes, show=False, color=c, area_alpha=0.5)
	fig.set_label(trig)
axes.figure.set_size_inches((20,20))
axes.figure.savefig("psd.png")
