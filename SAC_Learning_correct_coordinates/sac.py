import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from actor import DiagGaussianActor
from critic import DoubleQCritic
import copy
import os


import utils


class SACAgent(object):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device,discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, hidden_dim, hidden_depth, log_std_bounds):

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.log_std_bounds = log_std_bounds
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.critic = DoubleQCritic(obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim,hidden_depth=self.hidden_depth).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim=self.obs_dim,action_dim=self.action_dim,hidden_dim=self.hidden_dim, 
                                       hidden_depth=self.hidden_depth,log_std_bounds=self.log_std_bounds).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step):
        obs, action, next_obs, reward, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
            
    def mode(self, mode='train'):
         if mode == 'train':
            self.actor.train()
            self.critic.train()
         if mode == 'test':
            self.actor.eval()
            self.critic.eval()

    def save(self, model_dir, model_name):
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(self.critic.state_dict(), os.path.join(model_dir, f"{model_name}_critic.pth"))
                    torch.save(self.critic_optimizer.state_dict(), os.path.join(model_dir, f"{model_name}_critic_optimizer.pth"))
                    torch.save(self.actor.state_dict(), os.path.join(model_dir, f"{model_name}_actor.pth"))
                    torch.save(self.actor_optimizer.state_dict(), os.path.join(model_dir, f"{model_name}_actor_optimizer.pth"))

    
    def load_model(self, actor_path, critic_path, optim_a_path, optim_c_path):
                    try:
                        # Define the device to load the model to
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        if os.path.exists(actor_path):
                            # self.actor_target.load_state_dict(torch.load(actor_path, map_location=device))
                            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
                            print("Actor model loaded successfully")
                        else:
                            print(f"Actor model file not found: {actor_path}")

                        if os.path.exists(critic_path):
                            self.critic_target.load_state_dict(torch.load(critic_path, map_location=device))
                            self.critic.load_state_dict(torch.load(critic_path, map_location=device))
                            print("Critic model loaded successfully")
                        else:
                            print(f"Critic model file not found: {critic_path}")

                        if os.path.exists(optim_a_path):
                            self.actor_optimizer.load_state_dict(torch.load(optim_a_path, map_location=device))
                            print("Actor optimizer loaded successfully")
                        else:
                            print(f"Actor optimizer file not found: {optim_a_path}")

                        if os.path.exists(optim_c_path):
                            self.critic_optimizer.load_state_dict(torch.load(optim_c_path, map_location=device))
                            print("Critic optimizer loaded successfully")
                        else:
                            print(f"Critic optimizer file not found: {optim_c_path}")

                    except Exception as e:
                        print(f"Error loading model: {str(e)}")
                            

		
