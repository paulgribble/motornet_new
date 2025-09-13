import motornet as mn
import matplotlib.pyplot as plt
import numpy as np

plotor = mn.plotor.plot_pos_over_time


def plot_handpaths(episode_data, figtext=""):
    target_x = episode_data['targets'][:, -1, 0]
    target_y = episode_data['targets'][:, -1, 1]
    xy = episode_data['xy'][:,:,0:2].detach().numpy()
    fig, ax = plt.subplots(figsize=(5,3))
    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy)
    ax.scatter(target_x, target_y)
    fig.suptitle(f"{figtext}", fontsize=14)
    fig.tight_layout()
    return fig, ax

def plot_kinematics(episode_data, figtext=""):
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
    fig.suptitle(f"{figtext}", fontsize=14)
    fig.tight_layout()
    return fig, ax

def plot_activation(episode_data, figtext=""):
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
    fig.suptitle(f"{figtext}", fontsize=14)
    fig.tight_layout()
    return fig, ax

def window_average(x, w=10):
    rows = int(np.size(x)/w)  # round to (floor) int
    cols = w
    xw = x[0:w*rows].reshape((rows, cols)).mean(axis=1)
    return xw

def plot_losses(losses, figtext=""):
    n = len(losses.keys())
    fig, ax = plt.subplots(n,1,figsize=(6,12))
    for i,loss in enumerate(losses.keys()):
        x = range(len(losses[loss]))
        ax[i-1].semilogy(x, losses[loss])
        ax[i-1].set_ylabel(loss)
    ax[n-1].set_xlabel(f"Batch")
    fig.suptitle(f"{figtext}", fontsize=14)
    fig.tight_layout()
    return fig, ax


#  Plot arm trajectories and errors relative to the target
def plot_simulations(episode_data, figtext="", xylim=None):
    xy = episode_data['xy'][:,:,0:2].detach().numpy()
    target_xy = episode_data['targets'].detach().numpy()
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]
    fg = plt.figure(figsize=(10,4))
    fg.suptitle(figtext, fontsize=14)
    # trajectory in workspace
    plt.subplot(1,2,1)
    plt.ylim([-0.3, 1])
    plt.xlim([-0.7, 0.7])
    if xylim is not None:
        plt.xlim(xylim[0])
        plt.ylim(xylim[1])
    plotor(axis=plt.gca(), cart_results=xy)
    plt.scatter(target_x, target_y)
    # deviation from target
    plt.subplot(1,2,2)
    plt.ylim([-0.5, 0.5])
    plt.xlim([-0.5, 0.5])
    plotor(axis=plt.gca(), cart_results=xy - target_xy)
    plt.axhline(0, c="grey")
    plt.axvline(0, c="grey")
    plt.xlabel("X distance to target")
    plt.ylabel("Y distance to target")
    plt.show()


# Plot neural activity, muscle activity, inputs, targets and trajectory for a given episode
def plot_episode(episode_data, figtext=""):
    xy = episode_data['xy']
    all_targets = episode_data['targets']
    all_hidden = episode_data['hidden']
    all_muscle = episode_data['muscle']
    inp = episode_data['inp']
    fg, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    fg.suptitle(f'{figtext}', fontsize=14)
    ind = 0
    ax[0, 0].plot(np.squeeze(all_hidden.detach().cpu().numpy()[ind, :, :]))
    ax[0, 0].title.set_text('Neural activity')
    ax[0, 1].plot(np.squeeze(all_muscle.detach().cpu().numpy()[ind, :, :]))
    ax[0, 1].title.set_text('Muscle activity')
    ax[1, 0].plot(np.squeeze(inp.detach().cpu().numpy()[ind, :, :]))
    ax[1, 0].title.set_text('Inputs')
    ax[1, 1].plot(
        np.concatenate((all_targets.detach().cpu().numpy()[ind, :, 0:2], xy.detach().cpu().numpy()[ind, :, 0:2]),
                        axis=1))
    ax[1, 1].title.set_text('Targets and outputs')
    plt.show()
