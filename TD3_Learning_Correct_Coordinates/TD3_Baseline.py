import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

##### Implementation of the Actor Network
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

#### Implementation of the Critic Network
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim=7,
		action_dim=6,
		max_action=0.2,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

    #### Selection of Action from the State
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

    ### Learning Function for Actor and Critic Network
	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    #### Saving the Model
	# def save(self, model_dir, model_name):
	# 	torch.save(self.critic.state_dict(),model_dir+model_name+"_critic.pth")
	# 	torch.save(self.critic_optimizer.state_dict(), model_dir+model_name+"_critic_optimizer.pth")
		
	# 	torch.save(self.actor.state_dict(), model_dir+model_name+"_actor.pth")
	# 	torch.save(self.actor_optimizer.state_dict(), model_dir+model_name+"_actor_optimizer.pth")

	def save(self, model_dir, model_name):
		# Ensure the directory exists
		import os
		os.makedirs(model_dir, exist_ok=True)

		# Save the model components
		torch.save(self.critic.state_dict(), os.path.join(model_dir, f"{model_name}_critic.pth"))
		torch.save(self.critic_optimizer.state_dict(), os.path.join(model_dir, f"{model_name}_critic_optimizer.pth"))
		torch.save(self.actor.state_dict(), os.path.join(model_dir, f"{model_name}_actor.pth"))
		torch.save(self.actor_optimizer.state_dict(), os.path.join(model_dir, f"{model_name}_actor_optimizer.pth"))


	def load_model(self, actor_path, critic_path, optim_a_path, optim_c_path):
			try:
				# Define the device to load the model to
				device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

				if os.path.exists(actor_path):
					self.actor_target.load_state_dict(torch.load(actor_path, map_location=device))
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


	def mode(self, mode='train'):
         if mode == 'train':
            self.actor.train()
            self.critic.train()
         if mode == 'test':
            self.actor.eval()
            self.critic.eval()
		