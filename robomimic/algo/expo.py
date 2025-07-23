from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.loss_utils as LossUtils
from robomimic.utils.replay_buffer import ReplayBuffer

from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo, algo_factory, DiffusionPolicyUNet
from robomimic.models.policy_nets import PerturbationActorNetwork


@register_algo_factory_func("expo")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the TD3_BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    # only one variant of TD3_BC for now
    return Expo, {}


class Expo(PolicyAlgo, ValueAlgo):
    """
    Default Expo training
    """
    def __init__(self, **kwargs):
        PolicyAlgo.__init__(self, **kwargs)
        self.replay_buffer = ReplayBuffer(
            capacity=self.algo_config.replay_buffer.capacity,
            obs_key_shapes=self.obs_key_shapes,
            action_dim=self.ac_dim,
        )

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self._create_base_policy()
        self._create_edit_policy()
        self._create_critic()
        
        with torch.no_grad():
            TorchUtils.hard_update(
                source=self.critic, 
                target=self.critic_target,
            )

        # convert to float and move to device. base policy is already moved to device in DiffusionPolicyUNet
        self.edit_policy = self.edit_policy.float().to(self.device)
        self.critic = self.critic.float().to(self.device)
        self.critic_target = self.critic_target.float().to(self.device)

    def reset(self):
        self.base_policy.reset()

    def set_eval(self):
        self.base_policy.set_eval()
        self.edit_policy.eval()
        self.critic.eval()
        self.critic_target.eval()

    def _create_base_policy(self):
        """
        Creates the base policy network. Using DP for base policy
        TODO: Check how to upload pretrained model
        """
        self.base_policy = DiffusionPolicyUNet(
            algo_config=self.algo_config.base_policy,
            obs_config=self.obs_config,
            global_config=self.algo_config.base_policy,
            obs_key_shapes=self.obs_key_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
        )
    
    def _create_edit_policy(self):
        """
        Creates the action edit policy network.
        TODO: Action edit policy implementation should change. it should get base action as input
        """
        edit_policy_class = PerturbationActorNetwork
        edit_policy_args = dict(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.edit_policy.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.edit_policy = edit_policy_class(**edit_policy_args)
    
    def _create_critic(self):
        """
        Creates the critic network.
        """
        critic_class = ValueNets.ActionValueNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            value_bounds=self.algo_config.critic.value_bounds,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # Q network ensemble and target ensemble
        self.critic = nn.ModuleList()
        self.critic_target = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            critic = critic_class(**critic_args)
            self.critic.append(critic)

            critic_target = critic_class(**critic_args)
            self.critic_target.append(critic_target)

    def _get_target_q_values(self, obs_dict, action, goal_dict):
        """
        Helper function to get Q-values for a given observation and action.
        TODO: Make this parallelized
        """
        with torch.no_grad():
            q_values = []
            for target_critic in self.critic_target:
                q_value = target_critic(obs_dict, action, goal_dict)
                q_values.append(q_value)
            q_values = torch.cat(q_values, dim=0)
            return q_values.min()

    def get_action(self, obs_dict, goal_dict=None):
        """
        Based on EXPO algorithm, sample base action + action edit action. Select the hightest Q-value action out of samples

        Args:
            obs_dict (dict): dictionary of observations
            goal_dict (dict): dictionary of goals

        Returns:
            action (torch.Tensor): action
        """
        assert self.algo_config.on_the_fly_samples == 1, "sampling more than 1 action is not implemented" # TODO: Implement this
        
        # sample base and action edit actions
        base_actions = self.base_policy.get_action(obs_dict, goal_dict) # [N, ac_dim]
        edit_policy_obs_dict = copy.deepcopy(obs_dict)
        example_key = list(obs_dict.keys())[0]
        if len(obs_dict[example_key].shape) == 3:
            edit_policy_obs_dict = {obs_key: obs_dict[obs_key][:, -1, :] for obs_key in obs_dict}
        edit_actions = self.edit_policy(edit_policy_obs_dict, base_actions, goal_dict) # [N, ac_dim]
        action_samples = base_actions + edit_actions # [N, ac_dim]

        # select the action with the highest Q-value
        q_values = torch.stack([
            self._get_target_q_values(edit_policy_obs_dict, action_sample.unsqueeze(0), goal_dict)
            for action_sample in action_samples
        ], dim=0)  # Shape: [N]
        action_index = torch.argmax(q_values).item()
        return action_samples[action_index:action_index+1]
    
    def _get_target_values(self, next_states, goal_states, rewards, dones):
        """
        Helper function to get target values for training Q-function with TD-loss.

        Args:
            next_states (dict): batch of next observations
            goal_states (dict): if not None, batch of goal observations
            rewards (torch.Tensor): batch of rewards - should be shape (B, 1)
            dones (torch.Tensor): batch of done signals - should be shape (B, 1)

        Returns:
            q_targets (torch.Tensor): target Q-values to use for TD loss
        """

        with torch.no_grad():
            ########################
            # TODO: Implement this
            ########################
            q_targets = None

        return q_targets

    def _compute_critic_loss(self, critic, states, actions, goal_states, q_targets):
        """
        Helper function to compute loss between estimated Q-values and target Q-values.

        Nearly the same as BCQ (return type slightly different).

        Args:
            critic (torch.nn.Module): critic network
            states (dict): batch of observations
            actions (torch.Tensor): batch of actions
            goal_states (dict): if not None, batch of goal observations
            q_targets (torch.Tensor): batch of target q-values for the TD loss

        Returns:
            critic_loss (torch.Tensor): critic loss
        """
        q_estimated = critic(states, actions, goal_states)
        if self.algo_config.critic.use_huber:
            critic_loss = nn.SmoothL1Loss()(q_estimated, q_targets)
        else:
            critic_loss = nn.MSELoss()(q_estimated, q_targets)
        return critic_loss

    def _train_critic_on_batch(self, batch, epoch, no_backprop=False):
        """
        A modular helper function that can be overridden in case
        subclasses would like to modify training behavior for the
        critics.

        Exactly the same as BCQ (except for removal of @action_sampler_outputs and @critic_outputs)

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            no_backprop (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        info = OrderedDict()

        ########################
        # TODO: Impliment this function
        ########################

        print("train_critic_on_batch")
        return info
    
    def _train_base_policy_on_batch(self, batch, epoch, no_backprop=False):
        """
        TODO: Implement this function
        """
        print("train_base_policy_on_batch")
        return OrderedDict()
    
    def _train_edit_policy_on_batch(self, batch, epoch, no_backprop=False):
        """
        TODO: Implement this function
        """
        print("train_edit_policy_on_batch")
        return OrderedDict()
        
    def train_on_batch(self, epoch, validate=False):
        """
        Trains the policy and critic networks on a batch of data.
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, None, epoch, validate=validate) # simple checking

            ########################
            # TODO: Collect rollouts
            # We can take this out of this function and put it in online_rl.py
            # We will treat it as offline RL for now
            ########################

            # Critic training
            ########################
            # TODO: Train critic
            # Sample actions and use bellman backup to train critic
            # Iterate G times
            ########################
            for _ in range(self.algo_config.critic.n_iter):
                mini_batch = self.replay_buffer.sample_mini_batch(self.algo_config.critic.batch_size)
                critic_info = self._train_critic_on_batch(
                    batch=mini_batch, 
                    epoch=epoch, 
                    no_backprop=validate,
                )
                info.update(critic_info)
            ########################

            ########################
            # TODO: Update pi_base and pi_edit
            # For pi_base, use the last mini-batch with supervised learning
            # For pi_edit, use the last mini-batch maximizing objective Q - alpha * log (pi_edit)
            ########################
            last_mini_batch = self.replay_buffer.last_mini_batch(self.algo_config.critic.batch_size)
            base_policy_info = self._train_base_policy_on_batch(
                batch=last_mini_batch, 
                epoch=epoch, 
                no_backprop=validate,
            )
            edit_policy_info = self._train_edit_policy_on_batch(
                batch=last_mini_batch, 
                epoch=epoch, 
                no_backprop=validate,
            )
            info.update(base_policy_info)
            info.update(edit_policy_info)
