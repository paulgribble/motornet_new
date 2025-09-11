import motornet as mn
import matplotlib.pyplot as plt
import numpy as np

plotor = mn.plotor.plot_pos_over_time


def plot_handpaths(episode_data, figtext=None):
    target_x = episode_data['targets'][:, -1, 0]
    target_y = episode_data['targets'][:, -1, 1]
    xy = episode_data['xy'][:,:,0:2].detach().numpy()
    fig, ax = plt.subplots(figsize=(5,3))
    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy)
    ax.scatter(target_x, target_y)
    fig.tight_layout()
    return fig, ax

