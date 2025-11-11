import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.common.utils import *
from agents.common.buffers import *
from agents.common.networks import *


class Agent(object):
   """
   An implementation of the Proximal Policy Optimization (PPO) (by clipping) agent, 
   with early stopping based on approximate KL.
   """

   def __init__(self,
                env,
                args,
                device,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                lam=0.97,
                hidden_sizes=(64,64),
                sample_size=2048,
                train_policy_iters=80,
                train_vf_iters=80,
                minibatch_size=64,
                clip_param=0.2,
                target_kl=0.01,
                policy_lr=3e-4,
                vf_lr=1e-3,
                eval_mode=False,
                policy_losses=list(),
                vf_losses=list(),
                kls=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_limit = act_limit
      self.steps = steps 
      self.gamma = gamma
      self.lam = lam
      self.hidden_sizes = hidden_sizes
      self.sample_size = sample_size
      self.train_policy_iters = train_policy_iters
      self.train_vf_iters = train_vf_iters
      self.minibatch_size=minibatch_size
      self.clip_param = clip_param
      self.target_kl = target_kl
      self.policy_lr = policy_lr
      self.vf_lr = vf_lr
      self.eval_mode = eval_mode
      self.policy_losses = policy_losses
      self.vf_losses = vf_losses
      self.kls = kls
      self.logger = logger
      
      # === [FIX 1] ===
      # Store seed and first_episode flag
      self.seed = args.seed
      self.first_episode = True
      # ===============

      # Main network
      self.policy = GaussianPolicy(self.obs_dim, self.act_dim, self.act_limit, output_activation=F.tanh).to(self.device)
      self.vf = MLP(self.obs_dim, 1, activation=torch.tanh).to(self.device)
      
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.device, self.gamma, self.lam)

   def compute_vf_loss(self, obs, ret, v_old):
      # Prediction V(s)
      v = self.vf(obs).squeeze(1)

      # Value loss
      clip_v = v_old + torch.clamp(v-v_old, -self.clip_param, self.clip_param)
      vf_loss = torch.max(F.mse_loss(v, ret), F.mse_loss(clip_v, ret)).mean()
      return vf_loss

   def compute_policy_loss(self, obs, act, adv, log_pi_old):
      # Prediction logπ(s)
      _, _, _, log_pi = self.policy(obs, act, use_pi=False)
      
      # Policy loss
      ratio = torch.exp(log_pi - log_pi_old)
      clip_adv = torch.clamp(ratio, 1.-self.clip_param, 1.+self.clip_param)*adv
      policy_loss = -torch.min(ratio*adv, clip_adv).mean()

      # A sample estimate for KL-divergence, easy to compute
      approx_kl = (log_pi_old - log_pi).mean()
      return policy_loss, approx_kl

   def train_model(self):
      batch = self.buffer.get()
      obs = batch['obs']
      act = batch['act']
      ret = batch['ret']
      adv = batch['adv']
      
      # Prediction logπ_old(s), V_old(s)
      _, _, _, log_pi_old = self.policy(obs, act, use_pi=False)
      log_pi_old = log_pi_old.detach()
      v_old = self.vf(obs).squeeze(1)
      v_old = v_old.detach()

      batch_size = obs.shape[0]
      minibatch_size = self.minibatch_size if self.minibatch_size > 0 else batch_size

      # Train policy with multiple steps of gradient descent using minibatches
      for _ in range(self.train_policy_iters):
         permutation = torch.randperm(batch_size, device=self.device)
         for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = permutation[start:end]
            policy_loss, _ = self.compute_policy_loss(
               obs[mb_idx], act[mb_idx], adv[mb_idx], log_pi_old[mb_idx]
            )
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

         with torch.no_grad():
            _, _, _, log_pi = self.policy(obs, act, use_pi=False)
            kl = (log_pi_old - log_pi).mean()
            
         if kl > 1.5 * self.target_kl: # Early stopping at step i due to reaching max kl
            break

      # Train value function with multiple steps of gradient descent using minibatches
      for _ in range(self.train_vf_iters):
         permutation = torch.randperm(batch_size, device=self.device)
         for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = permutation[start:end]
            vf_loss = self.compute_vf_loss(obs[mb_idx], ret[mb_idx], v_old[mb_idx])
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

      # Save losses
      self.policy_losses.append(policy_loss.item())
      self.vf_losses.append(vf_loss.item())
      self.kls.append(kl.item())

   def run(self, max_step):
      step_number = 0
      total_reward = 0.
      info = {}

      # === [FIX 2] ===
      # Handle seeding and new reset() return values
      if self.first_episode:
         obs, info = self.env.reset(seed=self.seed)
         self.first_episode = False
      else:
         obs, info = self.env.reset()
      
      # Use new gymnasium done flags
      terminated = False
      truncated = False
      # ===============

      # Keep interacting until agent reaches a terminal state.
      # === [FIX 3] ===
      # Update loop condition
      while not (terminated or truncated or step_number == max_step):
      # ===============
         if self.args.render:
            self.env.render() # Note: render() is now typically handled at gym.make()

         if self.eval_mode:
            action, _, _, _ = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            
            # === [FIX 4] ===
            # Update env.step() return values
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            # ===============
         else:
            self.steps += 1
            
            # Collect experience (s, a, r, s') using some policy
            _, _, action, _ = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            
            # === [FIX 5] ===
            # Update env.step() return values
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Combine flags to get 'done' for buffer
            done = terminated or truncated
            # ===============

            # Add experience to buffer
            v = self.vf(torch.Tensor(obs).to(self.device))
            self.buffer.add(obs, action, reward, done, v.item())            
            
            # Start training when the number of experience is equal to sample size
            if self.steps == self.sample_size:
               self.buffer.finish_path()
               self.train_model()
               self.steps = 0
         total_reward += reward
         step_number += 1
         obs = next_obs
      
      if isinstance(info, dict):
         episode_info = info.get("episode")
         raw_episode_return = None
         if isinstance(episode_info, dict):
            raw_episode_return = episode_info.get("raw_return", None)
         if raw_episode_return is None:
            raw_episode_return = info.get("raw_episode_return", None)
         if raw_episode_return is not None:
            total_reward = raw_episode_return
         elif isinstance(episode_info, dict) and "r" in episode_info:
            total_reward = episode_info["r"]
         self.logger['RawReturn'] = float(raw_episode_return) if raw_episode_return is not None else float(total_reward)
      else:
         self.logger['RawReturn'] = float(total_reward)
      # Save logs
      if len(self.policy_losses) > 0:
         self.logger['LossPi'] = round(np.mean(self.policy_losses), 5)
         self.logger['LossV'] = round(np.mean(self.vf_losses), 5)
         self.logger['KL'] = round(np.mean(self.kls), 5)
      else:
         # Log 0.0 if no training has occurred in this episode
         self.logger['LossPi'] = 0.0
         self.logger['LossV'] = 0.0
         self.logger['KL'] = 0.0
      return step_number, total_reward
