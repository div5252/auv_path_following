#!/usr/bin/env python
from math import acos, sqrt
import rospy
import numpy as np
from uuv_control_interfaces import DPControllerBase
from torch import nn
import torch
from collections import deque
import random
import os
import os.path

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=100
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000
ACTION_SPACE_SIZE = 360

L = 10
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH = os.path.join(dir_path, '../params/rl_controller.pt')

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

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

        if os.path.isfile(PATH) and os.stat(PATH).st_size > 0:
            self.online_net.load_state_dict(torch.load(PATH))
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)

        self._num_iterations = 0

        self._trajectory = []

        self.vehicle_prev_pos = np.asarray([1, 0, 0])
        self.end_point = None

    def _reset_controller(self):
        super(RLController, self)._reset_controller()
        torch.save(self.online_net.state_dict(), PATH)
        rospy.signal_shutdown('episode over')

    def perpendicular_point(self, x):
        closest_dist = 1e10
        closest_index = -1
        l = 0
        r = len(self._trajectory) - 1
        if r < 0:
            return closest_index

        while(r>=l):
            mid = int((l+r)/2)
            dist = abs(self._trajectory[mid][0] - x)
            if dist < closest_dist:
                closest_dist = dist
                closest_index = mid
            if self._trajectory[mid][0] < x:
                l = mid + 1
            else:
                r = mid - 1
        return closest_index

    def norm(self, x):
        mod = sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        if mod == 0:
            return 1
        return mod
    
    def normalize(self, x):
        mod = self.norm(x)
        x = x / mod
        return x
    
    def euclidean_distance(self, x, y):
        return sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)
    
    def compute_state_reward(self):
        perpendicular_index = self.perpendicular_point(self._vehicle_model.pos[0])
        if perpendicular_index != -1:
            perpendicular_point = self._trajectory[perpendicular_index]
            perpendicular_point = np.asarray(perpendicular_point)
            d = self.euclidean_distance(self._vehicle_model.pos, perpendicular_point)

            slope = np.array([0, 0, 0])
            if perpendicular_index >= 1:
                slope = (perpendicular_point - self._trajectory[perpendicular_index - 1]) /  self._dt
            elif perpendicular_index + 1 < len(self._trajectory):
                slope = (self._trajectory[perpendicular_index + 1] - perpendicular_point) /  self._dt
            slope = self.normalize(slope)
            target = perpendicular_point + slope * L

            if self.end_point is not None and perpendicular_point[0] == self.end_point[0] and \
                perpendicular_point[1] == self.end_point[1] and perpendicular_point[2] == self.end_point[2]:
                target = self.end_point

            direction = target - self._vehicle_model.pos

            vehicle_angle = (self._vehicle_model.pos - self.vehicle_prev_pos) / self._dt
            
            diff_vector = np.dot(direction, vehicle_angle) / (self.norm(direction) * self.norm(vehicle_angle))
            diff_angle = acos(diff_vector)

            state = np.array([d, diff_angle], dtype=np.float32)
            rew = -0.9 * abs(diff_angle) + 0.1 * pow(2, 2 - (d / 10))

            return (state, rew)

        else:
            state = np.array([0, 0, 0])
            rew = 0.4
            return (state, rew)

    def compute_action(self, angle):
        tau = np.zeros(6)
        # Forward thrust
        tau[0] = 50 
        # Response to upward thrust
        tau[2] = -260
        # Rudder angle torque
        angle -= 180
        tau[5] = (float) (angle) / 180
        tau[5] *= 100
        return tau

    def check_done(self):
        if self.end_point is not None:
            dist = self.euclidean_distance(self._vehicle_model.pos, self.end_point)
            if dist <= 1:
                return True
        return False
    
    def update_controller(self):
        self._trajectory.append(self._reference['pos'])

        length = len(self._trajectory)
        if self.end_point is None and length >= 2 and self._trajectory[length - 1][0] == self._trajectory[length - 2][0] \
            and self._trajectory[length - 1][1] == self._trajectory[length - 2][1] and self._trajectory[length - 1][2] == self._trajectory[length - 2][2]:
            self.end_point = self._trajectory[length - 1]

        if self._num_iterations < MIN_REPLAY_SIZE:
            action = random.randint(0, ACTION_SPACE_SIZE - 1)
            obs, _ = self.compute_state_reward()

            # Perform action
            tau = self.compute_action(action)
            self.publish_control_wrench(tau)

            new_obs, rew = self.compute_state_reward()
            done = self.check_done()

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                self._reset_controller()

        else:
            step = self._num_iterations - MIN_REPLAY_SIZE
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

            rnd_sample = random.random()

            obs, _ = self.compute_state_reward()
            if rnd_sample <= epsilon:
                action = random.randint(0, ACTION_SPACE_SIZE - 1)
            else:
                action = self.online_net.act(obs)

            # Perform action
            tau = self.compute_action(action)
            self.publish_control_wrench(tau)

            new_obs, rew = self.compute_state_reward()
            done = self.check_done()

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)

            self.episode_reward += rew

            # Logging
            if step % 100 == 0:
                print('Step', step)
                print('Rew', rew)

            if done:
                self._reset_controller()
                self.rew_buffer.append(self.episode_reward)
            
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
        
        self._num_iterations += 1
        self.vehicle_prev_pos = self._vehicle_model.pos
        

if __name__ == '__main__':
    rospy.init_node('rl_controller_node')

    try:
        node = RLController()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
    print('exiting')