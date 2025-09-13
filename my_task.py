import numpy as np
import torch as th

# Here we are defining the Task. This is what we update to change the tasks the network has to learn.
class MyTask:
    def __init__(self, effector, **kwargs):
        self.effector = effector
        self.dt = self.effector.dt
        self.tgt_delay_range = kwargs.get('tgt_delay_range', [0.1, 0.5]) # tgt delay period
        self.go_delay_range  = kwargs.get('go_delay_range',  [0.5, 0.9]) # tgt delay period
        self.run_mode = kwargs.get('run_mode', 'train') # run mode – this is useful to switch between a training mode and an experimental mode, or different versions of a task

    def generate(self, batch_size, n_timesteps, **kwargs):
        # Draw the goal states randomly from all reachable states
        goal_states = self.effector.joint2cartesian(self.effector.draw_random_uniform_states(batch_size=batch_size)).to("cpu").numpy()
        # tile over time
        targets = np.tile(np.expand_dims(goal_states, axis=1), (1, n_timesteps, 1))
        # create empty input matrix with 2 dimensions for the x and y coordinates, and 1 for the go cue
        inputs = np.zeros(shape=(batch_size, n_timesteps, 2 + 1)) # target x,y and go cue

        tgt_delay_range = self.tgt_delay_range
        go_delay_range  = self.go_delay_range

        base_joint = np.deg2rad([50., 90., 0., 0.]).astype(np.float32)

        # Circular targets
        rad = 0.10
        n = 8
        angle = np.linspace(0,2*np.pi,n, endpoint=False)
        offset = rad*np.array([np.cos(angle), np.sin(angle),np.zeros(n),np.zeros(n)])

        if self.run_mode == 'test_center_out': # This is example of why alternate run modes are useful. We can turn off catch trials, fix the delay period length, and put the arm at one location
            catch_chance = 0.
            tgt_delay_range = [0.30, 0.30]
            go_delay_range  = [0.60, 0.60]
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
        elif self.run_mode == 'train_center_out':
            catch_chance = 0.5
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
        else: # train random
            catch_chance = 0.5 # catch trials with no go cue are useful to prevent the network from anticipating the go cue
            init_states = self.effector.draw_random_uniform_states(batch_size).detach().cpu().numpy() # random initial state

        # Create inputs and targets for all individual trials in the batch
        for i in range(batch_size):
            tgt_delay_time = generate_delay_time(tgt_delay_range[0] / self.dt, tgt_delay_range[1] / self.dt, 'random')
            go_delay_time  = generate_delay_time(go_delay_range[0]  / self.dt, go_delay_range[1]  / self.dt, 'random')
            start_point = self.effector.joint2cartesian(th.tensor(init_states[i,:])).to("cpu").numpy()

            if np.greater_equal(np.random.rand(), catch_chance):
                is_catch = False
            else:
                is_catch = True

            # Targets
            if self.run_mode == 'test_center_out':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + offset[:,i % n].T, axis = 1), (1, n_timesteps, 1))
            elif self.run_mode == 'train_center_out':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + offset[:,i % n].T, axis = 1), (1, n_timesteps, 1))
            else:
                targets[i, :, :] = np.tile(np.expand_dims(self.effector.joint2cartesian(self.effector.draw_random_uniform_states(1)).detach().cpu().numpy(), axis=1),(1, n_timesteps, 1))

            if not is_catch:
                inputs[i, 0:tgt_delay_time, 0:2] = start_point[0, 0:2]  # RNN sees start location until tgt_delay
                inputs[i, tgt_delay_time:,  0:2] = targets[i, -1, 0:2]  # then RNN sees final movement target
                inputs[i, 0:go_delay_time, 2] = 0  # RNN sees no-go until go_delay
                inputs[i, go_delay_time:,  2] = 1  # then RNN sees go
                targets[i, 0:go_delay_time, :] = start_point       # targets drive the loss function, desired xy is start_point until go_delay
                targets[i, go_delay_time:,  :] = targets[i, -1, :] # after go_delay desired xy is movement target position
            else:
                inputs[i, 0:tgt_delay_time, 0:2] = start_point[0, 0:2]  # RNN sees start location until tgt_delay
                inputs[i, tgt_delay_time:,  0:2] = targets[i, -1, 0:2]  # then RNN sees final movement target
                inputs[i, :,  2] = 0 # RNN sees no-go
                targets[i, :, :] = start_point # targets drive the loss function

            # Add noise to the inputs
            inputs[i, :, :] = inputs[i, :, :] + np.random.normal(loc=0., scale=1e-3,
                                                                 size=(inputs.shape[1], inputs.shape[2]))

        all_inputs = {"inputs": inputs}
        return [all_inputs, targets, init_states]
        #return [inputs, targets, init_states]

def generate_delay_time(delay_min, delay_max, delay_mode):
    if delay_mode == 'random':
        delay_time = np.random.uniform(delay_min, delay_max)
    elif delay_mode == 'noDelayInput':
        delay_time = 0
    else:
        raise AttributeError

    return int(delay_time)