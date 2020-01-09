from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_data_for_single_channel(data_for_channel, raw, picks):
    # raw.set_eeg_reference(ref_channels='average')
    # Set montage (location of channels)
    raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
    # Remove bad channels from analysis
    #raw.info['bads'] = ['F2', 'FFC2h', 'POO10h', 'O2']
    #picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
    pos = pos_from_raw(raw.info, picks)
    plot_single_topomap(picks, pos, title='', cmap_rb=True)


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
    plt.savefig('brain.png')
