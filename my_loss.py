import torch as th

def calculate_loss_mirzazadeh(episode_data):
    # Mirzazadeh Poune from Jonathan's lab
    # https://github.com/neural-control-and-computation-lab/MotorNet/tree/JAM-staging/MotorSaving

    xy      = episode_data['xy']
    targets = episode_data['targets']
    hidden  = episode_data['hidden']
    force   = episode_data['force']

    cartesian_loss = 1e+3 * th.mean(th.sum(th.abs(xy[:, :, 0:2] - targets), dim=-1))
    muscle_loss    = 1e-1 * th.mean(th.sum(force, dim=-1))
    spectral_loss  = 1e+4 * th.mean(th.sum(th.square(th.diff(hidden, 2, dim=1)), dim=-1))
    jerk_loss      = 1e+3 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))
    velocity_loss  = th.tensor(0e+0) #* th.mean(th.sum(th.square(xy[:, :, 2:]), dim=-1))
    activity_loss  = th.tensor(0e+0) #* th.mean(th.sum(th.square(hidden), dim=-1))

    total_loss = cartesian_loss + muscle_loss + velocity_loss + activity_loss + jerk_loss + spectral_loss

    return {
        'total'     : total_loss,
        'cartesian' : cartesian_loss,
        'muscle'    : muscle_loss,
        'velocity'  : velocity_loss,
        'activity'  : activity_loss,
        'spectral'  : spectral_loss,
        'jerk'      : jerk_loss
    }


def calculate_loss_michaels_2025_nature(episode_data):
    # from Sensory expectations shape neural population dynamics in motor circuits

    xy      = episode_data['xy']
    targets = episode_data['targets']
    hidden  = episode_data['hidden']
    force   = episode_data['force']

    cartesian_loss = 1e+3 * th.mean(th.sum(th.abs(xy[:, :, 0:2] - targets), dim=-1))
    muscle_loss    = 1e+0 * th.mean(th.sum(force, dim=-1))
    velocity_loss  = 2e+2 * th.mean(th.sum(th.square(xy[:, :, 2:]), dim=-1))
    activity_loss  = 1e-1 * th.mean(th.sum(th.square(hidden), dim=-1))
    spectral_loss  = 1e+4 * th.mean(th.sum(th.square(th.diff(hidden, 2, dim=1)), dim=-1))
    jerk_loss      = 1e+6 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))

    total_loss = cartesian_loss + muscle_loss + velocity_loss + activity_loss + jerk_loss + spectral_loss

    return {
        'total'     : total_loss,
        'cartesian' : cartesian_loss,
        'muscle'    : muscle_loss,
        'velocity'  : velocity_loss,
        'activity'  : activity_loss,
        'spectral'  : spectral_loss,
        'jerk'      : jerk_loss
    }


def calculate_loss_kashefi_2025(episode_data):
    # from Compositional neural dynamics during reaching

    xy      = episode_data['xy']
    targets = episode_data['targets']
    hidden  = episode_data['hidden']
    force   = episode_data['force']

    cartesian_loss = 1e+0 * th.mean(th.sum(th.abs(xy[:, :, 0:2] - targets), dim=-1))
    velocity_loss  = 1e-3 * th.mean(th.sum(th.square(xy[:, :, 2:]), dim=-1))
    jerk_loss      = 1e-4 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))
    muscle_loss    = 1e-4 * th.mean(th.sum(force, dim=-1))
    muscle_d_loss  = 1e-4 * th.mean(th.sum(th.square(th.diff(force, 1, dim=1)), dim=-1))
    activity_loss  = 1e-2 * th.mean(th.sum(th.square(hidden), dim=-1))
    spectral_loss  = 1e-1 * th.mean(th.sum(th.square(th.diff(hidden, 2, dim=1)), dim=-1))
  
    total_loss = cartesian_loss + velocity_loss + jerk_loss + muscle_loss + muscle_d_loss + activity_loss + spectral_loss

    return {
        'total'     : total_loss,
        'cartesian' : cartesian_loss,
        'velocity'  : velocity_loss,
        'jerk'      : jerk_loss,
        'muscle'    : muscle_loss,
        'muscle_d'  : muscle_d_loss,
        'activity'  : activity_loss,
        'spectral'  : spectral_loss,
    }


def calculate_loss_shahbazi_2025(episode_data):
    # from A Context-Free Model of Savings in Motor Learning

    xy      = episode_data['xy']
    targets = episode_data['targets']
    hidden  = episode_data['hidden']
    force   = episode_data['force']

    cartesian_loss = 1e+3 * th.mean(th.sum(th.abs(xy[:, :, 0:2] - targets), dim=-1))
    muscle_loss    = 1e-1 * th.mean(th.sum(force, dim=-1))
    spectral_loss  = th.tensor(0e+0) #* th.mean(th.sum(th.square(th.diff(hidden, 2, dim=1)), dim=-1))
    jerk_loss      = 1e+5 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))
    velocity_loss  = th.tensor(0e+0) #* th.mean(th.sum(th.square(xy[:, :, 2:]), dim=-1))
    activity_loss  = 1e-5 * th.mean(th.sum(th.square(hidden), dim=-1))

    total_loss = cartesian_loss + muscle_loss + velocity_loss + activity_loss + jerk_loss + spectral_loss

    return {
        'total'     : total_loss,
        'cartesian' : cartesian_loss,
        'muscle'    : muscle_loss,
        'velocity'  : velocity_loss,
        'activity'  : activity_loss,
        'spectral'  : spectral_loss,
        'jerk'      : jerk_loss
    }


def calculate_loss_paul(episode_data):
    # paul's fiddling around

    xy      = episode_data['xy']
    targets = episode_data['targets']
    hidden  = episode_data['hidden']
    force   = episode_data['force']

    position_loss = 1e+3 * th.mean(th.sum(th.abs(xy[:, :, 0:2] - targets), dim=-1))
    speed_loss    = 0e+0 * th.mean(th.sum(th.square(xy[:, :, 2:]), dim=-1))
    jerk_loss     = 0e+0 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))
    muscle_loss   = 1e+0 * th.mean(th.sum(force, dim=-1))
    muscle_d_loss = 0e+0 * th.mean(th.sum(th.square(th.diff(force, 1, dim=1)), dim=-1))
    hidden_loss   = 1e-1 * th.mean(th.sum(th.square(hidden), dim=-1))
    spectral_loss = 1e+4 * th.mean(th.sum(th.square(th.diff(hidden, 2, dim=1)), dim=-1))

    total_loss = position_loss + speed_loss + jerk_loss + muscle_loss + muscle_d_loss + hidden_loss + spectral_loss

    return {
        'total'     : total_loss,
        'position'  : position_loss,
        'speed'     : speed_loss,
        'jerk'      : jerk_loss,
        'muscle'    : muscle_loss,
        'muscle_d'  : muscle_d_loss,
        'hidden'    : hidden_loss,
        'spectral'  : spectral_loss,
    }
