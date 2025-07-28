"""
config for expo
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config import config_factory

class ExpoConfig(BaseConfig):
    ALGO_NAME = "expo"

    def train_config(self):
        """
        Setting up training parameters for Diffusion Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(ExpoConfig, self).train_config()
        
        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # set compatible data loading parameters
        self.train.seq_length = 16 # should match self.algo.base_policy.horizon.prediction_horizon
        self.train.frame_stack = 2 # should match self.algo.base_policy.horizon.observation_horizon
        
        # onlinerl parameters
        self.train.online_rollout_collection.n = 10
        self.train.n_iter.critic = 10
        self.train.n_iter.edit_policy = 1
        self.train.discount_factor = 0.99
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        # optimization parameters
        self.algo.optim_params.base_policy.optimizer_type = "adamw"
        self.algo.optim_params.base_policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.base_policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.base_policy.learning_rate.step_every_batch = True
        self.algo.optim_params.base_policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.base_policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        self.algo.optim_params.base_policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        self.algo.optim_params.base_policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.base_policy.regularization.L2 = 1e-6          # L2 regularization strength

        self.algo.optim_params.edit_policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.edit_policy.learning_rate.decay_factor = 0.0  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.edit_policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.edit_policy.regularization.L2 = 0.0          # L2 regularization strength

        self.algo.optim_params.critic.learning_rate.initial = 1e-4          # critic learning rate
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.0      # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.critic.learning_rate.epoch_schedule = []     # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = 0.00              # L2 regularization strength

        # base policy parameters
        self.algo.base_policy = config_factory("diffusion_policy").algo
        self.algo.base_policy_ckpt_path = "/home/hyunjun/projects/PreciseManip/sim/training/diffusion_policy_trained_models/DP_training/original/models/model_epoch_1500_low_dim_v15_success_0.18.pth"

        # edit policy parameters
        self.algo.edit_policy.net.type = "gaussian"
        self.algo.edit_policy.net.common.std_activation = "softplus"
        self.algo.edit_policy.net.common.low_noise_eval = True
        self.algo.edit_policy.net.common.use_tanh = False
        self.algo.edit_policy.net.gaussian.init_last_fc_weight = 0.001
        self.algo.edit_policy.net.gaussian.init_std = 0.3
        self.algo.edit_policy.net.gaussian.fixed_std = False
        self.algo.edit_policy.net.gmm.num_modes = 5
        self.algo.edit_policy.net.gmm.min_std = 0.0001
        self.algo.edit_policy.layer_dims = [300, 400]
        self.algo.edit_policy.max_gradient_norm = None
        self.algo.edit_policy.entropy_weight = 0.01
        self.algo.edit_policy.beta = 0.05

        # critic parameters
        self.algo.critic.ensemble.n = 2
        self.algo.critic.layer_dims = [300, 400]
        self.algo.critic.use_huber = False
        self.algo.critic.max_gradient_norm = None
        self.algo.critic.value_bounds = None
        self.algo.critic.target_tau = 0.01

        # replay buffer parameters
        self.algo.replay_buffer.capacity = 1000000
        
        # additional parameters
        self.algo.on_the_fly_samples = 1