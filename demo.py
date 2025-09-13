import os
import numpy as np
import torch as th
import motornet as mn
from tqdm import tqdm
import matplotlib.pyplot as plt

from my_env import MyEnvironment
from my_task import MyTask
from my_policy import create_policy
from my_utils import run_episode
import my_loss
from my_plots import plot_handpaths, plot_kinematics, plot_activation, plot_losses

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")

dt     = 0.010 # time step in seconds
ep_dur = 2.000 # episode duration in seconds

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

n_batches  = 10000
batch_size =    64
interval   =  1000

input_freeze  = 0      # don't freeze input weights
output_freeze = 0      # don't freeze output weights
optimizer_mod = 'Adam' # use the Adam optimizer
learning_rate = 1e-3   # set learning rate

policy, optimizer = create_policy(env, inputs, device, 
                                  policy_func   = mn.policy.ModularPolicyGRU, 
                                  optimizer_mod = optimizer_mod, 
                                  learning_rate = learning_rate)

loss_function = my_loss.calculate_loss_shahbazi_2025

task.run_mode = 'train' # random reaches

n_t = int(ep_dur / env.effector.dt) + 1 # number of time points

# make directory to store output
if not os.path.exists("output"):
    os.makedirs("output", exist_ok=True)

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
    
    loss_dict = loss_function(episode_data)
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
        fig.savefig(f"output/handpaths_{batch:04d}.png")
        plt.close(fig)
        fig,ax = plot_kinematics(episode_data, f"{batch:04d}")
        fig.savefig(f"output/kinematics_{batch:04d}.png")
        plt.close(fig)
        fig,ax = plot_activation(episode_data, f"{batch:04d}")
        fig.savefig(f"output/activation_{batch:04d}.png")
        plt.close(fig)


# test run on center-out task
task.run_mode = 'test_center_out'
episode_data = run_episode(env, task, policy, 8, n_t, device)

# plot the test
fig,ax = plot_handpaths(episode_data, "final")
fig.savefig("output/handpaths_final.png")
fig,ax = plot_kinematics(episode_data, "final")
fig.savefig("output/kinematics_final.png")
fig,ax = plot_activation(episode_data, "final")
fig.savefig("output/activation_final.png")


