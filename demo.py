import os
import pickle
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
from my_plots import plot_handpaths, plot_kinematics, plot_activation, plot_losses, plot_simulations, plot_episode

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")


dt         =    0.010 # time step in seconds
ep_dur     =    3.00  # episode duration in seconds
n_batches  = 2000
batch_size =   64
interval   =  100
output_dir = 'output'


mm = mn.muscle.RigidTendonHillMuscle()                    # muscle model
ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=dt) # effector model

# initialize the environment
env = MyEnvironment(max_ep_duration=ep_dur, effector=ee,
                    proprioception_delay=0.02, vision_delay=0.07,
                    proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4)
obs, info = env.reset()

# initialize the task
n_t  = int(ep_dur / env.effector.dt) + 1 # number of time points
task = MyTask(effector=env.effector)
inputs, targets, init_states = task.generate(1, n_t)

# simulation mode is "train" (random reaches) or "test" (8 center-out reaches)
sim_mode = "train"

optimizer_mod = 'Adam' # use the Adam optimizer
learning_rate = 1e-3   # set learning rate

policy, optimizer = create_policy(env, inputs, device, 
                                  policy_func   = mn.policy.ModularPolicyGRU, 
                                  optimizer_mod = optimizer_mod, 
                                  learning_rate = learning_rate)

loss_function = my_loss.calculate_loss_michaels_2025_nature

total_loss     = []
cartesian_loss = []
muscle_loss    = []
velocity_loss  = []
activity_loss  = []
spectral_loss  = []
jerk_loss      = []

# make directory to store output
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

print(f"Training: \n{n_batches} batches of \n{batch_size} simulations of \n{ep_dur} sec movements")

# training loop over batches
for batch in tqdm(iterable      = range(n_batches),
                  unit          = "batch",
                  total         = n_batches,
                  desc          = f"training {n_batches} batches of {batch_size}",
                  dynamic_ncols = True,
                  leave         = True):
    
    task.run_mode = 'train' # random reaches
    episode_data = run_episode(env, task, policy, batch_size, n_t, device)
    
    loss_dict = loss_function(episode_data)
    loss_dict['total'].backward()

    total_loss.append(     loss_dict['total'].item())
    cartesian_loss.append( loss_dict['cartesian'].item())
    muscle_loss.append(    loss_dict['muscle'].item())
    velocity_loss.append(  loss_dict['velocity'].item())
    activity_loss.append(  loss_dict['activity'].item())
    spectral_loss.append(  loss_dict['spectral'].item())
    jerk_loss.append(      loss_dict['jerk'].item())

    # important to make sure gradients don't get crazy
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1)  

    optimizer.step()
    optimizer.zero_grad()

    if (((batch % interval) == 0) and (batch > 0)):
        plot_simulations(episode_data, f"{batch:04d}")
        plot_episode(episode_data, f"{batch:04d}")

        # test run on center-out task
        task.run_mode = 'test_center_out'
        episode_data = run_episode(env, task, policy, 8, n_t, device)
        plot_simulations(episode_data, f"{batch:04d}", xylim=[[-.2,.1],[.3,.6]])
        plot_episode(episode_data, f"{batch:04d}")

        # Save weights
        th.save(policy.state_dict(), output_dir + f'/weights.pt')

losses = {
    'cartesian' : cartesian_loss,
    'muscle'    : muscle_loss,
    'velocity'  : velocity_loss,
    'activity'  : activity_loss,
    'spectral'  : spectral_loss,
    'jerk'      : jerk_loss
}

# test run on center-out task
task.run_mode = 'test_center_out'
episode_data = run_episode(env, task, policy, 8, n_t, device)

# plot the test
fig,ax = plot_handpaths(episode_data, "final")
fig.savefig(output_dir + "/handpaths_final.png")
fig,ax = plot_kinematics(episode_data, "final")
fig.savefig(output_dir + "/kinematics_final.png")
fig,ax = plot_activation(episode_data, "final")
fig.savefig(output_dir + "/activation_final.png")

# Save results
print(f"saving {output_dir}/weights.pt and {output_dir}/results.pt ...")
th.save(policy.state_dict(), output_dir + f'/weights.pt')
results = { 'episode_data': episode_data,
           'losses'      : losses
           }
th.save(results, output_dir + "/results.pt")


# Load weights and run center-out test
output_dir = "output"
w = th.load(output_dir + "/weights.pt", weights_only=True)
device = th.device("cpu")
dt     =    0.010 # time step in seconds
ep_dur =    3.00  # episode duration in seconds
mm = mn.muscle.RigidTendonHillMuscle()                    # muscle model
ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=dt) # effector model
env = MyEnvironment(max_ep_duration=ep_dur, effector=ee,
                    proprioception_delay=0.02, vision_delay=0.07,
                    proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4)
obs, info = env.reset()
n_t  = int(ep_dur / env.effector.dt) + 1 # number of time points
task = MyTask(effector=env.effector)
inputs, targets, init_states = task.generate(1, n_t)
policy, optimizer = create_policy(env, inputs, device, 
                                  policy_func   = mn.policy.ModularPolicyGRU, 
                                  optimizer_mod = 'Adam', 
                                  learning_rate = 3e-3)
policy.load_state_dict(w)
task.run_mode = 'test_center_out'
episode_data = run_episode(env, task, policy, 8, n_t, device)
plot_episode(episode_data)
plot_simulations(episode_data, xylim=[[-.2,.1],[.3,.6]])
plot_kinematics(episode_data)
plot_activation(episode_data)



