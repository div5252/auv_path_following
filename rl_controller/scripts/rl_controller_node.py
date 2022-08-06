#!/usr/bin/env python
from math import atan2
import rospy
import numpy as np
from uuv_control_interfaces import DPControllerBase
from torch import nn
import torch
from collections import deque
import random

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000
ACTION_SPACE_SIZE = 360

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Number of nodes in input layer
        in_features = 2

        # Discrete action space
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE))

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # Get q values from nn
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        # Detach tensor
        action = max_q_index.detach().item()

        return action


class RLController(DPControllerBase):
    def __init__(self):
        super(RLController, self).__init__(self)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=100)

        self.episode_reward = 0.0

        # Online network and target network
        self.online_net = Network()
        self.target_net = Network()

        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)

        self.num_iterations = 0

    def _reset_controller(self):
        super(RLController, self)._reset_controller()

    def compute_mechanics(self):
        d = self.error_pose_euler[2]
        L = self.error_pose_euler[0]
        c_d = atan2(d, L)
        error_angle = self.error_pose_euler[4]
        c = c_d - error_angle
        return (c, c_d, d)

    def compute_state(self):
        (c, _, d) = self.compute_mechanics()
        return (d, c)

    def compute_reward(self):
        (c, c_d, d) = self.compute_mechanics()
        rew = -0.9 * abs(c_d - c) + 0.1 * pow(2, 2 - (d / 10))
        return rew

    def check_done(self):
        if self.error_pose_euler < 0.001:
            return True
        return False
    
    def update_controller(self):
        if self.num_iterations < MIN_REPLAY_SIZE:
            action = random.randint(0, ACTION_SPACE_SIZE - 1)
            obs = self.compute_state()

            # Perform action
            self.publish_control_wrench([0, 0, 0])

            new_obs = self.compute_state()
            rew = self.compute_reward()
            done = self.check_done()

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                self._reset_controller()

        else:
            if self.num_iterations == MIN_REPLAY_SIZE:
                self._reset_controller()
            
            step = self.num_iterations - MIN_REPLAY_SIZE
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

            rnd_sample = random.random()

            obs = self.compute_state()
            if rnd_sample <= epsilon:
                action = random.randint(0, ACTION_SPACE_SIZE - 1)
            else:
                action = self.online_net.act(obs)

            # Perform action
            self.publish_control_wrench([0, 0, 0])

            new_obs = self.compute_state()
            rew = self.compute_reward()
            done = self.check_done()

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)

            self.episode_reward += rew

            if done:
                self._reset_controller()

                self.rew_buffer.append(self.episode_reward)
                self.episode_reward = 0.0
            
            # Start gradient step
            transitions = random.sample(self.replay_buffer, BATCH_SIZE)

            obses = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rews = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            new_obses = np.asarray([t[4] for t in transitions])

            obses_t = torch.as_tensor(obses, dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

            # Compute targets
            target_q_values = self.target_net(new_obses_t)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

            # Compute loss
            q_values = self.online_net(obses_t)

            action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update target network
            if step % TARGET_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # Logging
            if step % 1000 == 0:
                print('Step', step)
                print('Avg rew', np.mean(self.rew_buffer))
        
        self.num_iterations += 1
        

if __name__ == '__main__':
    rospy.init_node('rl_controller_node')

    try:
        node = RLController()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
    print('exiting')