"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.downsample_utils as DownsampleUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.diffusion_utils as DiffusionUtils
import robomimic.utils.smc_utils as SMCUtils

@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()

class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        if self.algo_config.language_conditioned:
            self.obs_shapes["lang_emb"] = [768] # clip is 768-dim embedding

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        self._is_goal_conditioned = False
        if self.goal_shapes is not None and len(self.goal_shapes) > 0:
            assert isinstance(self.goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': obs_encoder,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
            step_kwargs = {}
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
            step_kwargs = {
                "eta": self.algo_config.ddim.get("eta", 0.0)
            }
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            power = self.algo_config.ema.get("power", 0.75)
            min_decay = self.algo_config.ema.get("min_decay", 0.0)
            try:
                # Try newer version of diffusers that requires parameters
                parameters = []
                for net in nets.values():
                    if hasattr(net, 'parameters'):
                        parameters.extend(list(net.parameters()))
                ema = EMAModel(model=nets, parameters=parameters, power=power, min_decay=min_decay)
            except TypeError:
                # Fall back to older version of diffusers that doesn't require parameters
                ema = EMAModel(model=nets, power=power, min_value=min_decay)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.step_kwargs = step_kwargs
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.sample_action_queue = None

        # If using SMC
        if "smc" in self.algo_config and self.algo_config.smc.enabled:
            self.num_particles = self.algo_config.smc.num_particles
            self.resample_fn = SMCUtils.resampling_function(
                resample_strategy=self.algo_config.smc.resample_strategy,
                ess_threshold=self.algo_config.smc.ess_threshold,
                verbose=self.algo_config.smc.verbose
            )
            self.tempering_gamma = self.algo_config.smc.tempering_gamma
            self.kl_coeff = self.algo_config.smc.kl_coeff
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            ### YW: provide slack for range
            # in_range = (-1 <= actions) & (actions <= 1)
            # in_range = (-1.00001 <= actions) & (actions <= 1.00001)

            ### MPC can handle actions outside of [-1,1]
            in_range = (-1.1 <= actions) & (actions <= 1.1)

            ### YW
            all_in_range = torch.all(in_range).item()
            if torch.isnan(actions).any():
                import pdb; pdb.set_trace()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True

        ### YW: make sure actions are within [-1,1]. it's sometimes slightly off
        input_batch["actions"] = torch.clamp(input_batch["actions"], -1, 1)
        ### YW

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch['actions'].shape[0]

        # for k in batch["obs"]:
        #     print(f"{k}: {batch['obs'][k][0]}")
        # for k in batch["goal_obs"]:
        #     print(f"{k}: {batch['goal_obs'][k][0]}")

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch['actions']

            # encode obs
            inputs = {
                'obs': batch["obs"],
                'goal': batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                # print(k)
                # print(inputs['obs'][k].shape)
                # print(self.obs_shapes[k])
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
            for k in self.goal_shapes:
                # first dimension should be [B] for inputs
                assert inputs['goal'][k].ndim - 1 == len(self.goal_shapes[k])
                # repeat goal at temporal dimension
                inputs['goal'][k] = inputs['goal'][k].unsqueeze(1).repeat_interleave(inputs['obs'][k].shape[1], dim=1)
            obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)

            # predict the noise residual
            noise_pred = self.nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = F.mse_loss(noise_pred, noise)

            # logging
            losses = {
                'l2_loss': loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)

                step_info = {
                    'policy_grad_norms': policy_grad_norms
                }
                info.update(step_info)

        return info
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        sample_action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        self.sample_action_queue = sample_action_queue
        
        # Initialize previous gripper state
        self.wait_for_gripper = 0
        self.previous_gripper_state = 0
    
    def get_action(self, obs_dict, goal_dict=None, stride=1):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        action_queue_out = self.get_action_queue(obs_dict, goal_dict, stride=1)
        action = action_queue_out[:1] # [1, Da]

        return action
    
    def get_action_queue(self, obs_dict, goal_dict=None, stride=1, min_length=1, execution_length=1):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal
            stride (int): number of actions to skip
            min_length (int): minimum number of actions to return (used for path planning in TOPPRA)
            execution_length (int): number of actions to execute (used for execution after TOPPRA)

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)        

        if self.wait_for_gripper > 0:
            print(f"Waiting for gripper to change state: {self.wait_for_gripper}")
            if len(self.action_queue) == 0:
                action_sequence = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict)
                self.action_queue.extend(action_sequence[0])

            self.wait_for_gripper -= 1
            return self.action_queue.popleft().unsqueeze(0)
        
        assert execution_length <= min_length, "execution_length must be less than or equal to min_length"
        
        if len(self.action_queue) < stride * min_length:
            # if self.action_queue:
            #     self.action_queue.clear() # TODO temporal ensemble / RTC (inpainting)
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            
            # run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict)
            
            self.action_queue.extend(action_sequence[0, len(self.action_queue):])

        action_queue_out = torch.stack(list(self.action_queue))[stride-1::stride]
        
        for _ in range(stride * execution_length):
            self.action_queue.popleft()
           
            # print(f"Previous gripper state: {self.previous_gripper_state}")
            # print(f"Action gripper state: {int(action[9] > 0)}")
            # print(f"Gripper state changed: {(int(action[9] > 0) != self.previous_gripper_state)}")
            # if int(action[9] > 0) != self.previous_gripper_state:
            #     self.wait_for_gripper = 0
            #     break

        # self.previous_gripper_state = int(action[9] > 0)

        return action_queue_out
    
    def sample_action(self, obs_dict, goal_dict=None, num_samples=1, stride=1):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)        
        
        if len(self.sample_action_queue) < stride:
            if self.sample_action_queue:
                self.sample_action_queue.clear()
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            
            # run inference
            # [num_samples,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict, num_samples=num_samples)
            
            self.sample_action_queue.extend(action_sequence.transpose(0, 1)) # [T, num_samples, Da]
        
        for _ in range(stride):
            # has action, execute from left to right
            # [Da]
            action = self.sample_action_queue.popleft() # [num_samples, Da]

        return action
    
    def fill_sample_action_queue(self, obs_dict, goal_dict=None, num_samples=1):
        """
        Fill the sample action queue with num_samples actions.
        """
        assert not self.nets.training

        action_sequence = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict, num_samples=num_samples)
        self.sample_action_queue.extend(action_sequence.transpose(0, 1)[len(self.sample_action_queue):]) # [T, num_samples, Da]

        # self.sample_action_queue.clear()
        # self.sample_action_queue.extend(action_sequence.transpose(0, 1))
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None, num_samples=1):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            'obs': obs_dict,
            'goal': goal_dict
        }

        ### YW: only [B] for inputs
        only_b = True
        for k in self.obs_shapes:
            only_b = only_b and inputs['obs'][k].ndim - 1 == len(self.obs_shapes[k])

        if only_b:
            # [B] -> [B, T]
            for k in self.obs_shapes:
                inputs['obs'][k] = inputs['obs'][k].unsqueeze(1)

        inputs["obs"] = {k: inputs["obs"][k] for k in self.obs_shapes}
        ### YW

        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        for k in self.goal_shapes:
            # first dimension should be [B] for inputs
            assert inputs['goal'][k].ndim - 1 == len(self.goal_shapes[k])
            # repeat goal at temporal dimension
            inputs['goal'][k] = inputs['goal'][k].unsqueeze(1).repeat_interleave(inputs['obs'][k].shape[1], dim=1)
        obs_features = TensorUtils.time_distributed(inputs, nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)
        obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B * num_samples, Tp, action_dim), device=self.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
                **self.step_kwargs
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action
    
    def get_action_smc(self, obs_dict, goal_dict=None, reward_fn=None, guidance=True, base_action=None, skip_timesteps=0):
        assert not self.nets.training
        if skip_timesteps > 0:
            assert base_action is not None, "base_action must be provided if skip_timesteps is greater than 0"

        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        ac_start = To - 1
        ac_end = ac_start + Ta
        action_dim = self.ac_dim
        assert Ta == 1, "SMC only supports single action horizon"
        
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            'obs': obs_dict,
            'goal': goal_dict
        }

        ### YW: only [B] for inputs
        only_b = True
        for k in self.obs_shapes:
            only_b = only_b and inputs['obs'][k].ndim - 1 == len(self.obs_shapes[k])

        if only_b:
            # [B] -> [B, T]
            for k in self.obs_shapes:
                inputs['obs'][k] = inputs['obs'][k].unsqueeze(1)

        inputs["obs"] = {k: inputs["obs"][k] for k in self.obs_shapes}
        ### YW

        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        for k in self.goal_shapes:
            # first dimension should be [B] for inputs
            assert inputs['goal'][k].ndim - 1 == len(self.goal_shapes[k])
            # repeat goal at temporal dimension
            inputs['goal'][k] = inputs['goal'][k].unsqueeze(1).repeat_interleave(inputs['obs'][k].shape[1], dim=1) 
        obs_features = TensorUtils.time_distributed(inputs, nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)
        obs_cond = obs_cond.repeat_interleave(self.num_particles, dim=0)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B * self.num_particles, Tp, action_dim), device=self.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        nets['policy']['noise_pred_net'].requires_grad_(True)

        log_w = torch.zeros(B * self.num_particles, device=self.device)

        for i, k in enumerate(self.noise_scheduler.timesteps):
            if i < skip_timesteps:
                continue
            elif i == skip_timesteps and base_action is not None:
                naction = self.noise_scheduler.add_noise(base_action, naction, k)

            print(f"Timestep: {k}")
            prev_timestep = (
                k - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps
            )
            # to prevent OOB on gather
            prev_timestep = torch.clamp(prev_timestep, 0, self.noise_scheduler.config.num_train_timesteps - 1)

            if i > skip_timesteps:
                log_twist_func_prev = log_twist_func

            if guidance:
                with torch.enable_grad():
                    naction.requires_grad_(True)

                    # predict noise
                    noise_pred = nets['policy']['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # hat x_0
                    pred_original_sample = DiffusionUtils.ddim_prediction(
                        self.noise_scheduler,
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction,
                        **self.step_kwargs
                    )

                    # Predict Reward
                    rewards = reward_fn(pred_original_sample[:, ac_start:ac_end].squeeze(1))

                    # Reward Guidance (Approximation)
                    approx_guidance = torch.autograd.grad(outputs=rewards, inputs=naction, grad_outputs=torch.ones_like(rewards))[0]
                naction = naction.detach()
                rewards = rewards.detach()
                approx_guidance = approx_guidance.detach()
            else:
                naction.requires_grad_(True)

                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # hat x_0
                pred_original_sample = DiffusionUtils.ddim_prediction(
                    self.noise_scheduler,
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                    **self.step_kwargs
                )

                # Predict Reward
                rewards = reward_fn(pred_original_sample[:, ac_start:ac_end].squeeze(1))

                approx_guidance = torch.zeros_like(naction)
            
            print(f"Rewards: {rewards[:5]}")

            approx_guidance = torch.nan_to_num(approx_guidance) / self.kl_coeff

            # Tempering
            tempering_factor = min((1 + self.tempering_gamma)**i - 1, 1.)
            approx_guidance *= tempering_factor

            # Update log weights
            log_twist_func = tempering_factor * rewards / self.kl_coeff
            log_w += log_twist_func - log_twist_func_prev + log_prob_diffusion - log_prob_proposal if i>skip_timesteps else 0.

            # Resample if necessary
            resample_indices, is_resampled, new_log_w = self.resample_fn(log_w.detach().view(B, -1))
            log_w = new_log_w.view(-1).to(self.device)
            print(f"Resample Indices: {resample_indices}")

            naction = naction.view(B, self.num_particles, Tp, action_dim)[:, resample_indices].view(-1, Tp, action_dim)
            noise_pred = noise_pred.view(B, self.num_particles, Tp, action_dim)[:, resample_indices].view(-1, Tp, action_dim)

            # Propose next samples
            prev_sample, prev_sample_mean = DiffusionUtils.ddim_step_with_mean(
                self.noise_scheduler,
                model_output=noise_pred,
                timestep=k,
                sample=naction,
                **self.step_kwargs
            )
            variance = DiffusionUtils.get_variance(self.noise_scheduler, k, prev_timestep)
            # use eta = 1.
            # variance = self.algo_config.ddim.eta**2 * DiffusionUtils.left_broadcast(variance, prev_sample.shape).to(self.device)
            std_dev_t = variance.sqrt()
            naction = prev_sample + variance * approx_guidance

            # Calculate log_probs
            log_prob_diffusion = -0.5 * (naction - prev_sample_mean).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            log_prob_diffusion = log_prob_diffusion.sum(dim=(-2, -1)) 
            log_prob_proposal = -0.5 * (naction - prev_sample_mean - variance * approx_guidance).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            log_prob_proposal = log_prob_proposal.sum(dim=(-2, -1))

        # process action using Ta
        action = naction[:,ac_start:ac_end].squeeze(1)
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

    def set_train(self, freeze_image_encoder=False):
        """
        Set networks to train mode.
        """
        super(DiffusionPolicyUNet, self).set_train()
        if freeze_image_encoder:
            print("Freezing image encoder")
            self.nets["policy"]["obs_encoder"].eval()





# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            if h[-1].shape[-1] > 1:
                x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
