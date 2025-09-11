import torch as th


# Apply a curl force field
def applied_load(endpoint_vel, k, mode = 'CW'):
    # Curl Force Field
    if mode == 'CW':
        curl_matrix = th.tensor([[0., -1.], [1., 0.]])  # Clockwise
    else:
        curl_matrix = th.tensor([[0., 1.], [-1., 0.]])  # Counterclockwise
    force_field = k * endpoint_vel @ curl_matrix
    return force_field


# Run a single episode
def run_episode(env, task, policy, batch_size, n_t, device, k = 0):
    
    inputs, targets, init_states = task.generate(batch_size, n_t)
    targets = th.tensor(targets[:, :, 0:2], device=device, dtype=th.float)
    inp = th.tensor(inputs['inputs'], device=device, dtype=th.float)
    init_states = th.tensor(init_states, device=device, dtype=th.float)
    h = policy.init_hidden(batch_size)
    obs, info = env.reset(options={'batch_size': batch_size, 'joint_state': init_states})
    terminated = False

    # initialize things we want to keep track of
    xy = []
    all_actions = []
    all_muscle = []
    all_hidden = []
    all_force = []
    all_targets = []
    all_inp = []
    all_joint = []

    while not terminated:  # will run until `max_ep_duration` is reached
        t_step = int(env.elapsed / env.dt)
        obs = th.concat((obs, inp[:, t_step, :]), dim=1)
        action, h = policy(obs, h)

        # Compute endpoint load (force field)
        force_mask = (inp[:, t_step, 2].abs() < 1e-3).float().unsqueeze(1)
        force_field  = applied_load(endpoint_vel = info['states']['cartesian'][:, 2:], k = k, mode = 'CW')
        masked_force_field = force_field * force_mask

        obs,_,terminated,_,info = env.step(action=action, endpoint_load=masked_force_field)

        xy.append(info['states']['cartesian'][:, None, :])
        all_actions.append(action[:, None, :])
        all_muscle.append(info['states']['muscle'][:, 0, None, :])
        all_force.append(info['states']['muscle'][:, -1, None, :])
        all_hidden.append(h[:, None, :])
        all_targets.append(th.unsqueeze(targets[:, t_step, :], dim=1))
        all_inp.append(th.unsqueeze(inp[:, t_step, :], dim=1))
        all_joint.append(info['states']['joint'][:, None, :])

    return {
        'xy': th.cat(xy, dim=1),
        'hidden' : th.cat(all_hidden, dim=1),
        'actions' : th.cat(all_actions, dim=1),
        'muscle' : th.cat(all_muscle, dim=1),
        'force' : th.cat(all_force, dim=1),
        'targets' : th.cat(all_targets, dim=1),
        'inp' : th.cat(all_inp, dim=1),
        'joint' : th.cat(all_joint, dim=1)
    }

