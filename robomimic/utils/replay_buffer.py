import torch
import numpy as np
import h5py
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(
        self,
        capacity,
        obs_key_shapes,
        action_dim,
        observation_horizon=1,
        action_horizon=1,
    ):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        self.obs_key_shapes = obs_key_shapes
        self.action_dim = action_dim
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon

        self.action = torch.zeros((capacity, self.action_horizon, self.action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.obs = {obs_key: torch.zeros((capacity, self.observation_horizon, *self.obs_key_shapes[obs_key]), dtype=torch.float32) for obs_key in self.obs_key_shapes}
        self.next_obs = {obs_key: torch.zeros((capacity, self.observation_horizon, *self.obs_key_shapes[obs_key]), dtype=torch.float32) for obs_key in self.obs_key_shapes}

    def __len__(self):
        return self.size

    def initialize_buffer_from_hdf5(self, hdf5_path):
        hdf5_file = h5py.File(hdf5_path, "r")
        for ep in hdf5_file["data"]:
            obs = {obs_key: torch.from_numpy(hdf5_file["data/{}/obs/{}".format(ep, obs_key)][()]).float() for obs_key in self.obs_key_shapes}
            obs = stack_obs(obs, self.observation_horizon)
            next_obs = {obs_key: torch.from_numpy(hdf5_file["data/{}/next_obs/{}".format(ep, obs_key)][()]).float() for obs_key in self.obs_key_shapes}
            next_obs = stack_obs(next_obs, self.observation_horizon)
            actions = torch.from_numpy(hdf5_file["data/{}/actions".format(ep)][()]).float()
            rewards = torch.from_numpy(hdf5_file["data/{}/rewards".format(ep)][()]).float()[:, None]
            dones = torch.from_numpy(hdf5_file["data/{}/dones".format(ep)][()]).float()[:, None]
            self.insert_trajectory(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones
            )
        hdf5_file.close()

    def sample_mini_batch(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': {obs_key: self.obs[obs_key][idx].to(device) for obs_key in self.obs_key_shapes},
            'actions': self.action[idx].to(device),
            'rewards': self.rewards[idx].to(device),
            'next_obs': {obs_key: self.next_obs[obs_key][idx].to(device) for obs_key in self.obs_key_shapes},
            'dones': self.dones[idx].to(device)
        }

    def last_mini_batch(self, batch_size, device):
        idx = np.arange(self.size - batch_size, self.size)
        return {
            'obs': {obs_key: self.obs[obs_key][idx].to(device) for obs_key in self.obs_key_shapes},
            'actions': self.action[idx].to(device),
            'rewards': self.rewards[idx].to(device),
            'next_obs': {obs_key: self.next_obs[obs_key][idx].to(device) for obs_key in self.obs_key_shapes},
            'dones': self.dones[idx].to(device)
        }

    def insert_trajectory(self, obs, actions, rewards, next_obs, dones):
        batch_size = actions.shape[0]
        indices = (self.ptr + torch.arange(batch_size)) % self.capacity

        self.action[indices] = stack_action(actions, self.action_horizon)
        self.rewards[indices] = stack_reward(rewards, self.action_horizon)
        self.dones[indices] = stack_reward(dones, self.action_horizon)
        
        for obs_key in self.obs_key_shapes: # obs is already frame stacked when it is inserted
            self.obs[obs_key][indices] = obs[obs_key]

            # for next_obs, we should shift it according to prediction horizon
            next_obs_key = next_obs[obs_key]
            padding = next_obs_key[-1].unsqueeze(0).repeat(self.action_horizon - 1, 1, 1)
            next_obs_key = torch.cat([next_obs_key, padding], dim=0)
            self.next_obs[obs_key][indices] = next_obs_key[self.action_horizon-1:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)


def stack_action(action, action_horizon):
    """
    Frame stack actions
    """
    T, ac_dim = action.shape

    padding = action[-1].unsqueeze(0).repeat(action_horizon - 1, 1) # [action_horizon-1, ac_dim]
    padded_value = torch.cat([action, padding], dim=0)  # [T + action_horizon - 1, ac_dim]
    stacked_value = padded_value.unfold(dimension=0, size=action_horizon, step=1).permute(0, 2, 1)  # [T, action_horizon, ac_dim]
    return stacked_value

def stack_obs(obs, observation_horizon):
    """
    Frame stack observations
    """
    first_key = list(obs.keys())[0]
    T = obs[first_key].shape[0]

    new_obs = {}
    for key, value in obs.items():
        padding = value[0].unsqueeze(0).repeat(observation_horizon - 1, 1)  # [observation_horizon-1, obs_dim]
        padded_value = torch.cat([padding, value], dim=0)  # [T + observation_horizon - 1, obs_dim]
        stacked_value = padded_value.unfold(dimension=0, size=observation_horizon, step=1).permute(0, 2, 1)  # [T, observation_horizon, obs_dim]
        new_obs[key] = stacked_value
    return new_obs

def stack_reward(reward, action_horizon):
    reward = reward.float().unsqueeze(0).unsqueeze(0).squeeze(-1)  # shape: [1, 1, T]
    padding = torch.zeros((1, 1, action_horizon - 1), dtype=reward.dtype, device=reward.device) # [1, 1, action_horizon - 1]
    padded_reward = torch.cat([reward, padding], dim=2)  # [1, 1, T + action_horizon - 1]
    kernel = torch.ones(1, 1, action_horizon, dtype=torch.float32, device=reward.device)  # shape: [1, 1, action_horizon]
    conv_result = F.conv1d(padded_reward, kernel, stride=1, padding=0)  # shape: [1, 1, T]
    output = (conv_result.squeeze(0).squeeze(0) > 0).float().unsqueeze(-1)  # shape: [T, 1]
    return output