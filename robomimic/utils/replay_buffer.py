import torch
import numpy as np
import h5py

class ReplayBuffer:
    def __init__(
        self,
        capacity,
        obs_key_shapes,
        action_dim,
    ):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        self.obs_key_shapes = obs_key_shapes
        self.action_dim = action_dim

        self.action = torch.zeros((capacity, self.action_dim), dtype=torch.float32)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.obs = {obs_key: torch.zeros((capacity, *self.obs_key_shapes[obs_key]), dtype=torch.float32) for obs_key in self.obs_key_shapes}
        self.next_obs = {obs_key: torch.zeros((capacity, *self.obs_key_shapes[obs_key]), dtype=torch.float32) for obs_key in self.obs_key_shapes}

    def __len__(self):
        return self.size

    def initialize_buffer_from_hdf5(self, hdf5_path):
        hdf5_file = h5py.File(hdf5_path, "r")
        for ep in hdf5_file["data"]:
            self.batched_insert(
                obs={obs_key: torch.from_numpy(hdf5_file["data/{}/obs/{}".format(ep, obs_key)][()]).float() for obs_key in self.obs_key_shapes},
                action=torch.from_numpy(hdf5_file["data/{}/actions".format(ep)][()]).float(),
                reward=torch.from_numpy(hdf5_file["data/{}/rewards".format(ep)][()]).float()[:, None],
                next_obs={obs_key: torch.from_numpy(hdf5_file["data/{}/next_obs/{}".format(ep, obs_key)][()]).float() for obs_key in self.obs_key_shapes},
                done=torch.from_numpy(hdf5_file["data/{}/dones".format(ep)][()]).float()[:, None]
            )
        hdf5_file.close()

    def sample_mini_batch(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': {obs_key: self.obs[obs_key][idx] for obs_key in self.obs_key_shapes},
            'action': self.action[idx],
            'reward': self.reward[idx],
            'next_obs': {obs_key: self.next_obs[obs_key][idx] for obs_key in self.obs_key_shapes},
            'done': self.dones[idx]
        }

    def last_mini_batch(self, batch_size):
        idx = np.arange(self.size - batch_size, self.size)
        return {
            'obs': {obs_key: self.obs[obs_key][idx] for obs_key in self.obs_key_shapes},
            'action': self.action[idx],
            'reward': self.reward[idx],
            'next_obs': {obs_key: self.next_obs[obs_key][idx] for obs_key in self.obs_key_shapes},
            'done': self.dones[idx]
        }
    
    def batched_insert(self, obs, action, reward, next_obs, done):
        batch_size = action.shape[0]
        indices = (self.ptr + torch.arange(batch_size)) % self.capacity

        self.action[indices] = action
        self.reward[indices] = reward
        self.dones[indices] = done
        
        for obs_key in self.obs_key_shapes:
            self.next_obs[obs_key][indices] = next_obs[obs_key]
            self.obs[obs_key][indices] = obs[obs_key]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        for obs_key in self.obs_key_shapes:
            self.next_obs[obs_key][self.ptr] = next_obs[obs_key]
            self.obs[obs_key][self.ptr] = obs[obs_key]
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)