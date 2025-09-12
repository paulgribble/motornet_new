
import numpy as np
import torch as th
import motornet as mn
from tqdm import tqdm
import matplotlib.pyplot as plt

from my_env import MyEnvironment
from my_task import MyTask
from my_policy import create_policy
from my_utils import run_episode
from my_loss import calculate_loss_michaels_2025_nature
from my_plots import plot_handpaths, plot_kinematics, plot_activation

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")

dt     = 0.01 # time step in seconds
ep_dur = 1.30 # episode duration in seconds

mm = mn.muscle.RigidTendonHillMuscle()                    # muscle model
ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=dt) # effector model

# initialize the environment
env = MyEnvironment(max_ep_duration=ep_dur, effector=ee,
                    proprioception_delay=0.02, vision_delay=0.07,
                    proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4)
obs, info = env.reset()
n_t = int(ep_dur / env.effector.dt)

# initialize the task
task = MyTask(effector=env.effector)
inputs, targets, init_states = task.generate(1, n_t)

# simulation mode is "train" (random reaches) or "test" (8 center-out reaches)
sim_mode = "train"

n_batches  = 2000
batch_size =   64
interval   =  500

results = {}

input_freeze  = 0      # don't freeze input weights
output_freeze = 0      # don't freeze output weights
optimizer_mod = 'Adam' # use the Adam optimizer
learning_rate = 3e-3   # set learning rate

policy, optimizer = create_policy(env, inputs, device, 
                                  policy_func   = mn.policy.ModularPolicyGRU, 
                                  optimizer_mod = optimizer_mod, 
                                  learning_rate = learning_rate)

total_losses     = []
cartesian_losses = []
muscle_losses    = []
spectral_losses  = []
jerk_losses      = []
endpoint_dev     = []
lateral_dev      = []

task.run_mode = 'train' # random reaches

n_t = int(ep_dur / env.effector.dt) + 1 # number of time points

# training loop over batches
for batch in tqdm(iterable = range(n_batches),
                  unit          = "batch",
                  total         = n_batches,
                  mininterval   = 1.0,
                  desc          = f"training {n_batches} batches of {batch_size}",
                  dynamic_ncols = True,
                  leave         = True):
    
    task.run_mode = 'train' # random reaches
    episode_data = run_episode(env, task, policy, batch_size, n_t, device)
    
    loss_dict = calculate_loss_michaels_2025_nature(episode_data)
    total_losses.append(loss_dict['total'].item())
    cartesian_losses.append(loss_dict['cartesian'].item())
    muscle_losses.append(loss_dict['muscle'].item())
    spectral_losses.append(loss_dict['spectral'].item())
    jerk_losses.append(loss_dict['jerk'].item())

    loss_dict['total'].backward()

    # important to make sure gradients don't get crazy
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1)  

    optimizer.step()
    optimizer.zero_grad()

    if (((batch % interval) == 0) and (batch > 0)):
        # test run on center-out task
        task.run_mode = 'test_center_out'
        episode_data = run_episode(env, task, policy, 8, n_t, device)
        # plot the test
        fig,ax = plot_handpaths(episode_data, f"{batch:04d}")
        fig.savefig(f"handpaths_{batch:04d}.png")
        plt.close(fig)
        fig,ax = plot_kinematics(episode_data, f"{batch:04d}")
        fig.savefig(f"kinematics_{batch:04d}.png")
        plt.close(fig)
        fig,ax = plot_activation(episode_data, f"{batch:04d}")
        fig.savefig(f"activation_{batch:04d}.png")
        plt.close(fig)


# test run on center-out task
task.run_mode = 'test_center_out'
episode_data = run_episode(env, task, policy, 8, n_t, device)

# plot the test
fig,ax = plot_handpaths(episode_data, "final")
fig.savefig("handpaths_final.png")
fig,ax = plot_kinematics(episode_data, "final")
fig.savefig("kinematics_final.png")
fig,ax = plot_activation(episode_data, "final")
fig.savefig("activation_final.png")


