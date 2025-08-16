from collections import OrderedDict, deque
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

@register_algo_factory_func("dsrl")
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
    return DSRL, {}

class DSRL(PolicyAlgo, ValueAlgo):
    def __init__(self, **kwargs):
        PolicyAlgo.__init__(self, **kwargs)
        
        self.replay_buffer = ReplayBuffer(
            capacity=self.algo_config.replay_buffer.capacity,
            obs_key_shapes=self.obs_key_shapes,
            action_dim=self.ac_dim,
            observation_horizon=self.algo_config.base_policy.horizon.observation_horizon,
            action_horizon=self.algo_config.base_policy.horizon.action_horizon,
        )

    def _create_shapes(self, obs_keys, obs_key_shapes):
        """
        Create obs_shapes, goal_shapes, and subgoal_shapes dictionaries, to make it
        easy for this algorithm object to keep track of observation key shapes. Each dictionary
        maps observation key to shape.

        Args:
            obs_keys (dict): dict of required observation keys for this training run (usually
                specified by the obs config), e.g., {"obs": ["rgb", "proprio"], "goal": ["proprio"]}
            obs_key_shapes (dict): dict of observation key shapes, e.g., {"rgb": [3, 224, 224]}
        """
        # determine shapes
        self.obs_shapes = OrderedDict()
        self.goal_shapes = OrderedDict()
        self.subgoal_shapes = OrderedDict()

        # We check across all modality groups (obs, goal, subgoal), and see if the inputted observation key exists
        # across all modalitie specified in the config. If so, we store its corresponding shape internally
        for k in obs_key_shapes:
            if "obs" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.obs.values() for obs_key in modality]:
                self.obs_shapes[k] = [self.algo_config.base_policy.horizon.observation_horizon, *obs_key_shapes[k]]
            if "goal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.goal.values() for obs_key in modality]:
                self.goal_shapes[k] = obs_key_shapes[k]
            if "subgoal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.subgoal.values() for obs_key in modality]:
                self.subgoal_shapes[k] = obs_key_shapes[k]

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self._create_base_policy()
        self._create_dsrl_policy()
        self._create_base_critic()
        self._create_dsrl_critic()

        with torch.no_grad():
            TorchUtils.hard_update(
                source=self.nets["base_critic"],
                target=self.nets["base_critic_target"],
            )
            TorchUtils.hard_update(
                source=self.nets["dsrl_critic"],
                target=self.nets["dsrl_critic_target"],
            )

        self.nets.float().to(self.device)

    def _create_base_policy(self):
        """
        Creates the base policy network. Using DP for base policy
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

    def _create_dsrl_policy(self):
        """
        Creates the dsrl policy network.
        """
        dsrl_policy_args = dict(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim * self.algo_config.base_policy.horizon.prediction_horizon,
            mlp_layer_dims=self.algo_config.dsrl_policy.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            use_tanh=True,
            scale=self.algo_config.dsrl_policy.beta,
        )
        self.nets["dsrl_policy"] = PolicyNets.GaussianActorNetwork(**dsrl_policy_args)

    def _create_base_critic(self):
        """
        Creates the base critic network.
        """
        critic_class = ValueNets.ActionValueNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim * self.algo_config.base_policy.horizon.action_horizon,
            mlp_layer_dims=self.algo_config.base_critic.layer_dims,
            value_bounds=self.algo_config.base_critic.value_bounds,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # Q network ensemble and target ensemble
        self.nets["base_critic"] = nn.ModuleList()
        self.nets["base_critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.base_critic.ensemble.n):
            critic = critic_class(**critic_args)
            self.nets["base_critic"].append(critic)

            critic_target = critic_class(**critic_args)
            self.nets["base_critic_target"].append(critic_target)

        # Note: we optimize the log of the entropy coeff which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        init_value = 1.0
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=0.001)
        self.target_entropy = 0.0
    
    def _create_dsrl_critic(self):
        """
        Creates the dsrl critic network.
        """
        critic_class = ValueNets.ActionValueNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim * self.algo_config.base_policy.horizon.prediction_horizon,
            mlp_layer_dims=self.algo_config.dsrl_critic.layer_dims,
            value_bounds=self.algo_config.dsrl_critic.value_bounds,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # Q network ensemble and target ensemble
        self.nets["dsrl_critic"] = nn.ModuleList()
        self.nets["dsrl_critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.dsrl_critic.ensemble.n):
            critic = critic_class(**critic_args)
            self.nets["dsrl_critic"].append(critic)

            critic_target = critic_class(**critic_args)
            self.nets["dsrl_critic_target"].append(critic_target)

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        """
        Args:
            obs_dict (dict): dictionary of observations
                each obervation has N samples
            goal_dict (dict): dictionary of goals

        Returns:
            action (torch.Tensor): action
        """
        noisy_action = self.nets["dsrl_policy"](obs_dict, goal_dict) # [N, ac_dim * prediction_horizon]
        noisy_action = noisy_action.reshape(-1, self.algo_config.base_policy.horizon.prediction_horizon, self.ac_dim) # [N, prediction_horizon, ac_dim]
        action = self.base_policy._get_action_trajectory(obs_dict, goal_dict, noisy_action) # [N, prediction_horizon, ac_dim]
        return action
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.base_policy.horizon.observation_horizon
        Ta = self.algo_config.base_policy.horizon.action_horizon

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            # import pdb; pdb.set_trace()
            # obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.obs_queue))
            # obs_dict_tensor = dict((k, torch.cat(v, dim=0).unsqueeze(0)) for k,v in obs_dict_list.items())
            
            # run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action

    def train_one_epoch(self, epoch, validate=False):
        self.set_train()
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, None, epoch, validate=validate) # simple checking

            ########################
            # Train critic
            # Sample actions and use bellman backup to train critic
            # Iterate G times
            ########################
            print("Training base critic")
            for _ in LogUtils.custom_tqdm(range(self.global_config.train.n_iter.base_critic)):
                mini_batch = self.replay_buffer.sample_mini_batch(
                    self.global_config.train.batch_size,
                    self.device
                )
                critic_info = self._train_base_critic_on_batch(
                    batch=mini_batch, 
                    epoch=epoch, 
                    no_backprop=validate,
                )
                info.update(critic_info)

            print("Training dsrl critic")
            for _ in LogUtils.custom_tqdm(range(self.global_config.train.n_iter.dsrl_critic)):
                mini_batch = self.replay_buffer.sample_mini_batch(
                    self.global_config.train.batch_size,
                    self.device
                )
                critic_info = self._train_dsrl_critic_on_batch(
                    batch=mini_batch, 
                    epoch=epoch, 
                    no_backprop=validate,
                )
                info.update(critic_info)

            print("Training dsrl policy")
            for _ in LogUtils.custom_tqdm(range(self.global_config.train.n_iter.dsrl_policy)):
                mini_batch = self.replay_buffer.sample_mini_batch(
                    self.global_config.train.batch_size,
                    self.device
                )
                policy_info = self._train_dsrl_policy_on_batch(
                    batch=mini_batch, 
                    epoch=epoch, 
                    no_backprop=validate,
                )
                info.update(policy_info)

            return info
        
    def _get_base_critic_target_q_values(self, obs_dict, action, goal_dict):
        """
        Helper function to get Q-values for a given observation and action.

        Args:
            obs_dict (dict): dictionary of (N, Do) sized obervations
            action (torch.Tensor): (N, ac_dim) sized action
            goal_dict (dict): dictionary of (N, Dg) sized goals

        Returns:
            q_values (torch.Tensor): (N, ) sized Q-values
        """
        q_values = torch.cat([
            target_critic(obs_dict, action, goal_dict)
            for target_critic in self.nets["base_critic_target"]
        ], dim=1)
        return q_values.min(dim=1).values.unsqueeze(1)
        
    def _get_dsrl_critic_target_q_values(self, obs_dict, action, goal_dict):
        """
        Helper function to get Q-values for a given observation and action.
        """
        q_values = torch.cat([
            target_critic(obs_dict, action, goal_dict)
            for target_critic in self.nets["dsrl_critic_target"]
        ], dim=1)
        return q_values.min(dim=1).values.unsqueeze(1)
    
    def _update_entropy_coefficient_on_batch(self, batch, no_backprop=False):
        obs = batch["obs"]

        action_pred_dist = self.nets["dsrl_policy"].forward_train(obs, goal_dict=None)
        action_pred_samples = action_pred_dist.sample()
        action_pred_log_probs = action_pred_dist.log_prob(action_pred_samples)
        ent_coeff = torch.exp(self.log_ent_coef.detach())
        ent_coeff_loss = -(self.log_ent_coef * (action_pred_log_probs + self.target_entropy).detach()).mean()
        if not no_backprop:
            self.ent_coef_optimizer.zero_grad()
            ent_coeff_loss.backward()
            self.ent_coef_optimizer.step()
        return ent_coeff, ent_coeff_loss
        
    def _train_base_critic_on_batch(self, batch, epoch, no_backprop=False):
        info = OrderedDict()

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        next_obs = batch["next_obs"]

        # Update entropy coefficient
        ent_coeff, ent_coeff_loss = self._update_entropy_coefficient_on_batch(batch, no_backprop)
        info["base_critic/ent_coeff"] = ent_coeff
        info["base_critic/ent_coeff_loss"] = ent_coeff_loss

        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=None) for critic in self.nets["base_critic"]]

        with torch.no_grad():
            next_noise_dist = self.nets["dsrl_policy"].forward_train(next_obs, goal_dict=None)
            next_noise_samples = next_noise_dist.sample()
            next_noise_log_probs = next_noise_dist.log_prob(next_noise_samples).unsqueeze(1)
            next_noise_samples = next_noise_samples.reshape(-1, self.algo_config.base_policy.horizon.prediction_horizon, self.ac_dim)

            next_actions = self.base_policy._get_action_trajectory(next_obs, goal_dict=None, noisy_action=next_noise_samples)
            next_pred_q = self._get_base_critic_target_q_values(next_obs, next_actions, None)
            next_pred_q = next_pred_q - ent_coeff * next_noise_log_probs

            target_q = rewards + self.global_config.train.discount_factor * (1 - dones) * next_pred_q

        base_critic_losses = []
        loss_fn = nn.SmoothL1Loss() if self.algo_config.base_critic.use_huber else nn.MSELoss()
        for i, pred_q in enumerate(pred_qs):
            loss = loss_fn(pred_q, target_q)
            info[f"base_critic/loss_{i}"] = loss
            base_critic_losses.append(loss)
        
        if not no_backprop:
            for (base_critic_loss, base_critic, base_critic_target, optimizer) in zip(
                base_critic_losses, self.nets["base_critic"], self.nets["base_critic_target"], self.optimizers["base_critic"]
            ):
                TorchUtils.backprop_for_loss(
                    net=base_critic,
                    optim=optimizer,
                    loss=base_critic_loss,
                    max_grad_norm=self.algo_config.base_critic.max_gradient_norm,
                    retain_graph=False,
                )
                with torch.no_grad():
                    TorchUtils.soft_update(source=base_critic, target=base_critic_target, tau=self.algo_config.base_critic.target_tau)

        return info

    def _train_dsrl_critic_on_batch(self, batch, epoch, no_backprop=False):
        info = OrderedDict()

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        next_obs = batch["next_obs"]

        B = actions.shape[0]
        Tp = self.algo_config.base_policy.horizon.prediction_horizon
        Ta = self.algo_config.base_policy.horizon.action_horizon
        action_dim = self.ac_dim

        noisy_actions = torch.randn(
            (B, Tp, action_dim), device=self.device)
        
        actions = self.base_policy._get_action_trajectory(obs, None, noisy_actions)
        dsrl_qs = [critic(obs_dict=obs, acts=noisy_actions, goal_dict=None) for critic in self.nets["dsrl_critic"]]
        with torch.no_grad():
            base_qs = [critic(obs_dict=obs, acts=actions, goal_dict=None) for critic in self.nets["base_critic"]]

        dsrl_q_losses = []
        loss_fn = nn.SmoothL1Loss() if self.algo_config.dsrl_critic.use_huber else nn.MSELoss()
        for i, (dsrl_q, base_q) in enumerate(zip(dsrl_qs, base_qs)):
            loss = loss_fn(dsrl_q, base_q)
            info[f"dsrl_critic/loss_{i}"] = loss
            dsrl_q_losses.append(loss)

        if not no_backprop:
            for (dsrl_q_loss, dsrl_critic, dsrl_critic_target, optimizer) in zip(
                dsrl_q_losses, self.nets["dsrl_critic"], self.nets["dsrl_critic_target"], self.optimizers["dsrl_critic"]
            ):
                TorchUtils.backprop_for_loss(
                    net=dsrl_critic,
                    optim=optimizer,
                    loss=dsrl_q_loss,
                    max_grad_norm=self.algo_config.dsrl_critic.max_gradient_norm,
                    retain_graph=False,
                )
                with torch.no_grad():
                    TorchUtils.soft_update(source=dsrl_critic, target=dsrl_critic_target, tau=self.algo_config.dsrl_critic.target_tau)

        return info

    def _train_dsrl_policy_on_batch(self, batch, epoch, no_backprop=False):
        info = OrderedDict()

        obs = batch["obs"]

        # Update entropy coefficient
        ent_coeff, ent_coeff_loss = self._update_entropy_coefficient_on_batch(batch, no_backprop)
        info["dsrl_policy/ent_coeff"] = ent_coeff
        info["dsrl_policy/ent_coeff_loss"] = ent_coeff_loss

        noisy_actions_dist = self.nets["dsrl_policy"].forward_train(obs, goal_dict=None)
        noisy_actions = noisy_actions_dist.rsample()
        noisy_actions_log_probs = noisy_actions_dist.log_prob(noisy_actions).unsqueeze(1)
        noisy_actions = noisy_actions.reshape(-1, self.algo_config.base_policy.horizon.prediction_horizon, self.ac_dim)

        dsrl_qs = self._get_dsrl_critic_target_q_values(obs, noisy_actions, None)
        dsrl_loss = -dsrl_qs.mean() + ent_coeff * noisy_actions_log_probs.mean()
        info["dsrl_policy/loss"] = dsrl_loss
        info["dsrl_policy/q_pred"] = dsrl_qs.mean()

        if not no_backprop:
            TorchUtils.backprop_for_loss(
                net=self.nets["dsrl_policy"],
                optim=self.optimizers["dsrl_policy"],
                loss=dsrl_loss,
                max_grad_norm=self.algo_config.dsrl_policy.max_gradient_norm,
                retain_graph=False,
            )
        
        return info