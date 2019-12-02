from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_single_topomap(data, pos, vmin=None, vmax=None, title=None, cmap_rb=False):
    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax
    fig, ax = plt.subplots()
    cmap = mne.viz.utils._setup_cmap('interactive', norm=1 - cmap_rb)
    im, _ = mne.viz.topomap.plot_topomap(data, pos, vmin=vmin, vmax=vmax, axes=ax,
                                         cmap=cmap[0], image_interp='bilinear', contours=0,
                                         outlines='skirt', show=False)
    cbar, cax = mne.viz.topomap._add_colorbar(ax, im, cmap, pad=.25, title=None,
                                              size="10%", format='%3.3f')
    cbar.set_ticks((vmin, vmax))
    ax.set_title(title)
    # Load data, set montage and remove bad channels as before
    # Load data
    raw = mne.io.read_raw_brainvision('../data/20191104_Cybathlon_Test_1.vhdr')
    # Set montage (location of channels)
    raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
    montage = mne.channels.read_montage("standard_1020")
    raw.set_montage(montage)
    raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
    # Remove bad channels from analysis
    raw.info['bads'] = ['F2', 'FFC2h']
    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
    # Do something with your data per channel and get one number per channel as result (as 1D np array)
    some_data_per_channel = np.arange(124)
    pos = pos_from_raw(raw.info, picks)
    plot_single_topomap(some_data_per_channel, pos, title='')
