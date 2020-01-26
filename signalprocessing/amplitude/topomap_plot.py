from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import numpy as np
import matplotlib.pyplot as plt
import mne


# This script was provided by Anja Meunier for plotting a single Channel
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
    # plt.savefig('brain.png')
    plt.show()
