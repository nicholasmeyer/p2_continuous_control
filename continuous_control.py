import os
from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch

import matplotlib.pyplot as plt

from ddpg_agent import Agent

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment configuration
env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64", no_graphics=False)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# environment information
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents in the environment
n_agents = len(env_info.agents)
print('Number of agents:', n_agents)
# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


def load(agent):
    if os.path.isfile('checkpoint_actor.pth'):
        agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        agent.actor_target.load_state_dict(torch.load('checkpoint_actor.pth'))
    if os.path.isfile('checkpoint_critic.pth'):
        agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
        agent.critic_target.load_state_dict(torch.load('checkpoint_critic.pth'))


def ddpg_train(
        n_episodes = 1000,
        n_agents = n_agents,
        state_size = state_size,
        action_size = action_size,
        seed = 0,
        buffer_size = int(1e5),
        batch_size = 128,
        gamma = 0.99,
        tau = 1e-3,
        lr_actor = 1e-4,
        lr_critic = 1e-4,
        weight_decay = 0):
    scores = []
    scores_window = deque(maxlen=100)
    agent = Agent(
        n_agents,
        state_size,
        action_size,
        seed,
        buffer_size,
        batch_size,
        gamma,
        tau,
        lr_actor,
        lr_critic,
        weight_decay)
    load(agent)
    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations
        agent.reset()  # reset the agent noise
        score = np.zeros(n_agents)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards  # update the score
            states = next_states  # roll over the state to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        scores.append(np.mean(score))
        scores_window.append(np.mean(score))
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score),
                                                                                   np.mean(scores_window)), end="")
        if n_episodes % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if np.mean(scores_window) >= 30.0:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            break
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def trained_agent(n_agents,
        state_size,
        action_size,
        seed,
        buffer_size,
        batch_size,
        gamma,
        tau,
        lr_actor,
        lr_critic,
        weight_decay):

    agent = Agent(
        n_agents,
        state_size,
        action_size,
        seed,
        buffer_size,
        batch_size,
        gamma,
        tau,
        lr_actor,
        lr_critic,
        weight_decay)

    for episode in range(3):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(n_agents)
        while True:
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards
            states = next_states
            if np.any(dones):
                break
        print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))
    env.close()


if __name__ == "__main__":
    ddpg_train()
