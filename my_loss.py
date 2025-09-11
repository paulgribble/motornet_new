import torch as th

# Compute losses for a single episode.
def calculate_loss(episode_data):

    xy      = episode_data['xy']
    targets = episode_data['targets']
    hidden  = episode_data['hidden']
    force   = episode_data['force']

    cartesian_loss = 1e+3 * th.mean(th.sum(th.abs(xy[:, :, 0:2] - targets), dim=-1))
    muscle_loss    = 1e-1 * th.mean(th.sum(force, dim=-1))
    spectral_loss  = 1e+4 * th.mean(th.sum(th.square(th.diff(hidden, 2, dim=1)), dim=-1))
    jerk_loss      = 1e+3 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))

    total_loss = cartesian_loss + muscle_loss + jerk_loss + spectral_loss

    return {
        'total'     : total_loss,
        'cartesian' : cartesian_loss,
        'muscle'    : muscle_loss,
        'spectral'  : spectral_loss,
        'jerk'      : jerk_loss
    }


