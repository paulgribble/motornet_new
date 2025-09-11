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
    fig.suptitle(f'figtext', fontsize=16)
    fig.tight_layout()
    return fig, ax

def plot_kinematics(episode_data, figtext=None):
    all_xy = episode_data['xy'][:,:,0:2].detach().numpy()
    n = np.shape(all_xy)[0] # movements
    all_vel = episode_data['xy'][:,:,2:4].detach().numpy()
    all_tg = episode_data['targets'].detach().numpy()
    all_inp = episode_data['inp'][:,:,0:2].detach().numpy()
    x = np.linspace(0, np.shape(all_xy)[1], np.shape(all_xy)[1])
    fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(6, 10))
    for i in range(n):
        ax[i, 0].plot(x, np.array(all_inp[i, :, :]), ':')
        ax[i, 0].plot(x, np.array(all_tg[i, :, :]), '--')
        ax[i, 0].plot(x, np.array(all_xy[i, :, :]), '-')
        ax[i, 1].plot(x, np.array(all_vel[i, :, :]), '-')
        ax[i, 0].set_ylabel('xy,tg')
        ax[i, 1].set_ylabel('vel')
        ax[i, 0].set_xlabel('time steps')
        ax[i, 1].set_xlabel('time steps')
    fig.suptitle(f'figtext', fontsize=16)
    fig.tight_layout()
    return fig, ax

def plot_activation(episode_data, figtext=None):
    all_muscles = episode_data['actions'].detach().numpy()
    all_hidden = episode_data['hidden'].detach().numpy()
    n = np.shape(all_muscles)[0]
    nt = np.shape(all_muscles)[1]
    fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(6, 10))
    x = np.linspace(0, nt, nt)
    for i in range(n):
        ax[i, 0].plot(x, np.array(all_muscles[i, :, :]))
        ax[i, 1].plot(x, np.array(all_hidden[i, :, :]))
        ax[i, 0].set_ylabel('muscle act (au)')
        ax[i, 1].set_ylabel('hidden act (au)')
        ax[i, 0].set_xlabel('time steps')
        ax[i, 1].set_xlabel('time steps')
    fig.suptitle(f'figtext', fontsize=16)
    fig.tight_layout()
    return fig, ax
