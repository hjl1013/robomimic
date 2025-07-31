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
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.utils.replay_buffer import ReplayBuffer

from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo, algo_factory, DiffusionPolicyUNet


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
            observation_horizon=self.algo_config.base_policy.horizon.observation_horizon,
            action_horizon=self.algo_config.base_policy.horizon.action_horizon,
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
                source=self.nets["critic"], 
                target=self.nets["critic_target"],
            )

        # convert to float and move to device. base policy is already moved to device in DiffusionPolicyUNet
        self.nets["edit_policy"] = self.nets["edit_policy"].float().to(self.device)
        self.nets["critic"] = self.nets["critic"].float().to(self.device)
        self.nets["critic_target"] = self.nets["critic_target"].float().to(self.device)

    def reset(self):
        self.base_policy.reset()

    def _create_base_policy(self):
        """
        Creates the base policy network. Using DP for base policy
        TODO: Check how to upload pretrained model
        """
        assert self.algo_config.base_policy_ckpt_path is not None, "base_policy_ckpt_path is not set"
        self.base_policy = DiffusionPolicyUNet(
            algo_config=self.algo_config.base_policy,
            obs_config=self.obs_config,
            global_config=self.algo_config.base_policy,
            obs_key_shapes=self.obs_key_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
        )
        self.nets["base_policy"] = self.base_policy.nets
        ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=self.algo_config.base_policy_ckpt_path)
        self.base_policy.deserialize(ckpt_dict["model"])

    def _create_edit_policy(self):
        """
        Creates the action edit policy network.
        TODO: Action edit policy implementation should change. it should get base action as input
        """
        edit_policy_class = PolicyNets.GaussianActorNetwork
        edit_policy_obs_shapes = copy.deepcopy(self.obs_shapes)
        edit_policy_obs_shapes["base_action"] = (self.ac_dim, )
        edit_policy_args = dict(
            obs_shapes=edit_policy_obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.edit_policy.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets["edit_policy"] = edit_policy_class(**edit_policy_args)
    
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
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            critic = critic_class(**critic_args)
            self.nets["critic"].append(critic)

            critic_target = critic_class(**critic_args)
            self.nets["critic_target"].append(critic_target)

    def _get_target_q_values(self, obs_dict, action, goal_dict):
        """
        Helper function to get Q-values for a given observation and action.

        Args:
            obs_dict (dict): dictionary of (N, Do) sized obervations
            action (torch.Tensor): (N, ac_dim) sized action
            goal_dict (dict): dictionary of (N, Dg) sized goals

        Returns:
            q_values (torch.Tensor): (N, ) sized Q-values
        TODO: Make this parallelized
        """
        with torch.no_grad():
            q_values = torch.cat([
                target_critic(obs_dict, action, goal_dict)
                for target_critic in self.nets["critic_target"]
            ], dim=1)
            return q_values.min(dim=1).values.unsqueeze(1)

    def _get_q_values(self, obs_dict, action, goal_dict):
        """
        Helper function to get Q-values for a given observation and action.

        Args:
            obs_dict (dict): dictionary of (N, Do) sized obervations
            action (torch.Tensor): (N, ac_dim) sized action
            goal_dict (dict): dictionary of (N, Dg) sized goals

        Returns:
            q_values (torch.Tensor): (N, ) sized Q-values
        TODO: Make this parallelized
        """
        with torch.no_grad():
            q_values = torch.cat([
                critic(obs_dict, action, goal_dict)
                for critic in self.nets["critic"]
            ], dim=1)
            return q_values.min(dim=1).values.unsqueeze(1)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Based on EXPO algorithm, sample base action + action edit action. Select the hightest Q-value action out of samples
        We should consider diffusion with no action chunking.

        Args:
            obs_dict (dict): dictionary of observations
                each obervation has N samples
            goal_dict (dict): dictionary of goals

        Returns:
            action (torch.Tensor): action
        """
        # self.base_policy.set_eval()
        sample_num = self.algo_config.on_the_fly_samples

        # sample base actions
        duplicated_obs_dict = {obs_key: obs_dict[obs_key].repeat_interleave(sample_num, dim=0) for obs_key in obs_dict}
        base_actions = self.base_policy._get_action_trajectory(duplicated_obs_dict, goal_dict)[:, 0, :] # [N * sample_num, ac_dim] we assume no action chunking
        
        # sample action edits
        edit_policy_obs_dict = copy.deepcopy(duplicated_obs_dict)
        if self.global_config.train.frame_stack > 1:
            edit_policy_obs_dict = {obs_key: duplicated_obs_dict[obs_key][:, -1, :] for obs_key in duplicated_obs_dict} # action edit doesn't use frame stack
        edit_policy_obs_dict["base_action"] = base_actions
        edit_actions = self.nets["edit_policy"](edit_policy_obs_dict, goal_dict) # [N * sample_num, ac_dim]

        # combine base actions and action edits
        action_samples = base_actions + edit_actions # [N * sample_num, ac_dim]
        q_values = self._get_q_values(edit_policy_obs_dict, action_samples, goal_dict) # [N * sample_num, 1]
        
        # reshape action and q_values
        action_samples = action_samples.reshape(-1, sample_num, self.ac_dim) # [N, sample_num, ac_dim]
        q_values = q_values.reshape(-1, sample_num) # [N, sample_num]

        # select the action with the highest Q-value
        action_index = torch.argmax(q_values, dim=1)
        return action_samples[torch.arange(action_samples.shape[0]), action_index, :] # [N, ac_dim]

    def get_base_edit_action(self, obs_dict, goal_dict=None):
        """
        Based on EXPO algorithm, sample base action + action edit action.

        Args:
            obs_dict (dict): dictionary of observations
                each obervation has N samples
            goal_dict (dict): dictionary of goals

        Returns:
            action (torch.Tensor): action
        """
        # self.base_policy.set_eval()
        sample_num = self.algo_config.on_the_fly_samples

        # sample base actions
        duplicated_obs_dict = {obs_key: obs_dict[obs_key].repeat_interleave(sample_num, dim=0) for obs_key in obs_dict}
        base_actions = self.base_policy._get_action_trajectory(duplicated_obs_dict, goal_dict)[:, 0, :] # [N * sample_num, ac_dim] we assume no action chunking
        
        # sample action edits
        edit_policy_obs_dict = copy.deepcopy(duplicated_obs_dict)
        if self.global_config.train.frame_stack > 1:
            edit_policy_obs_dict = {obs_key: duplicated_obs_dict[obs_key][:, -1, :] for obs_key in duplicated_obs_dict} # action edit doesn't use frame stack
        edit_policy_obs_dict["base_action"] = base_actions
        edit_actions = self.nets["edit_policy"](edit_policy_obs_dict, goal_dict) # [N * sample_num, ac_dim]

        # combine base actions and action edits and calculate q-values
        action_samples = base_actions + edit_actions # [N * sample_num, ac_dim]
        q_values = self._get_q_values(edit_policy_obs_dict, action_samples, goal_dict) # [N * sample_num, 1]
        
        # reshape q_values and select the action with the highest Q-value
        q_values = q_values.reshape(-1, sample_num) # [N, sample_num]
        action_index = torch.argmax(q_values, dim=1)

        # reshape action samples
        base_actions = base_actions.reshape(-1, sample_num, self.ac_dim) # [N, sample_num, ac_dim]
        edit_actions = edit_actions.reshape(-1, sample_num, self.ac_dim) # [N, sample_num, ac_dim]

        return base_actions[torch.arange(base_actions.shape[0]), action_index, :], edit_actions[torch.arange(edit_actions.shape[0]), action_index, :] # [N, ac_dim]

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
        
        # get batch values
        obs = batch["obs"]
        actions = batch["actions"]
        next_obs = batch["next_obs"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # reshape obs and next_obs if frame_stack is greater than 1
        if self.global_config.train.frame_stack > 1:
            unstacked_obs = {obs_key: obs[obs_key][:, -1, :] for obs_key in obs}
            unstacked_next_obs = {obs_key: next_obs[obs_key][:, -1, :] for obs_key in next_obs}

        # Q predictions
        pred_qs = [critic(obs_dict=unstacked_obs, acts=actions, goal_dict=None)
                   for critic in self.nets["critic"]]

        # target Q value
        next_actions = self.get_action(next_obs)
        target_qs = self._get_target_q_values(unstacked_next_obs, next_actions, None)
        q_target = rewards + self.global_config.train.discount_factor * (1 - dones) * target_qs
        q_target = q_target.detach()

        # compute critic losses
        critic_losses = []
        td_loss_fcn = nn.SmoothL1Loss() if self.algo_config.critic.use_huber else nn.MSELoss()
        for (i, q_pred) in enumerate(pred_qs):
            # Calculate td error loss
            td_loss = td_loss_fcn(q_pred, q_target)
            info[f"critic/critic{i+1}_loss"] = td_loss
            critic_losses.append(td_loss)
        
        # update critic
        if not no_backprop:
            for (critic_loss, critic, critic_target, optimizer) in zip(
                critic_losses, self.nets["critic"], self.nets["critic_target"], self.optimizers["critic"]
            ):
                TorchUtils.backprop_for_loss(
                    net=critic,
                    optim=optimizer,
                    loss=critic_loss,
                    max_grad_norm=self.algo_config.critic.max_gradient_norm,
                    retain_graph=False,
                )
                with torch.no_grad():
                    TorchUtils.soft_update(source=critic, target=critic_target, tau=self.algo_config.critic.target_tau)

        return info
    
    def _train_base_policy_on_batch(self, batch, epoch, no_backprop=False):
        """
        TODO: Implement this function
        """
        print("train_base_policy_on_batch")
        raise NotImplementedError("train_base_policy_on_batch is not implemented")
        # info = OrderedDict()

        # batch["goal_obs"] = None
        # info_base_policy = self.base_policy.train_on_batch(batch, epoch, validate=no_backprop)
        # info = {f"base_policy/{key}": value for key, value in info_base_policy.items()}

        return info
    
    def _train_edit_policy_on_batch(self, batch, epoch, no_backprop=False):
        """
        TODO: Implement this function
        """
        info = OrderedDict()

        # get batch values
        obs = batch["obs"]

        # reshape obs and next_obs if frame_stack is greater than 1
        if self.global_config.train.frame_stack > 1:
            unstacked_obs = {obs_key: obs[obs_key][:, -1, :] for obs_key in obs}

        # calculate actions
        base_actions, _ = self.get_base_edit_action(obs, None)
        base_actions = base_actions.detach()

        # 1. prepare edit_policy input
        edit_policy_obs_dict = copy.deepcopy(unstacked_obs)  # s
        edit_policy_obs_dict["base_action"] = base_actions   # a

        # 2. sample edit actions (a_hat) and get distribution
        dist = self.nets["edit_policy"].forward_train(edit_policy_obs_dict, goal_dict=None)
        a_hat = dist.rsample()  # reparameterized sampling
        log_prob = dist.log_prob(a_hat)  # shape [B,]

        # 3. evaluate Q(s, a + a_hat)
        combined_action = base_actions + a_hat
        q_pred = self._get_q_values(unstacked_obs, combined_action, goal_dict=None)  # shape [B, 1]

        # 4. compute loss
        alpha = self.algo_config.edit_policy.entropy_weight  # entropy regularization
        actor_loss = (-q_pred + alpha * log_prob.unsqueeze(1)).mean()

        # 5. return info
        info["edit_policy/loss"] = actor_loss
        info["edit_policy/log_prob"] = log_prob.mean()
        info["edit_policy/q_pred"] = q_pred.mean()

        if not no_backprop:
            TorchUtils.backprop_for_loss(
                net=self.nets["edit_policy"],
                optim=self.optimizers["edit_policy"],
                loss=actor_loss,
                max_grad_norm=self.algo_config.edit_policy.max_gradient_norm,
                retain_graph=False,
            )

        return info
        
    def train_one_epoch(self, epoch, validate=False):
        """
        Trains the policy and critic networks on a batch of data.
        """
        self.set_train()

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, None, epoch, validate=validate) # simple checking


            ########################
            # Train critic
            # Sample actions and use bellman backup to train critic
            # Iterate G times
            ########################
            print("Training critic")
            for _ in LogUtils.custom_tqdm(range(self.global_config.train.n_iter.critic)):
                mini_batch = self.replay_buffer.sample_mini_batch(
                    self.global_config.train.batch_size,
                    self.device
                )
                critic_info = self._train_critic_on_batch(
                    batch=mini_batch, 
                    epoch=epoch, 
                    no_backprop=validate,
                )
                info.update(critic_info)

            ########################
            # Update pi_base and pi_edit
            # For pi_base, use the last mini-batch with supervised learning
            # For pi_edit, use the last mini-batch maximizing objective Q - alpha * log (pi_edit)
            ########################
            print("Training edit policy")
            for _ in LogUtils.custom_tqdm(range(self.global_config.train.n_iter.edit_policy)):
                last_mini_batch = self.replay_buffer.last_mini_batch(
                    self.global_config.train.batch_size,
                    self.device
                )
                # I think we don't have to update base policy. I will not implement this
                # base_policy_info = self._train_base_policy_on_batch(
                #     batch=last_mini_batch, 
                #     epoch=epoch, 
                #     no_backprop=validate,
                # )
                edit_policy_info = self._train_edit_policy_on_batch(
                    batch=last_mini_batch, 
                    epoch=epoch, 
                    no_backprop=validate,
                )
                # info.update(base_policy_info)
                info.update(edit_policy_info)

        return info