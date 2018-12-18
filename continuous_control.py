import os
import argparse
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from ddpg_agent import Agent

# environment configuration
env = UnityEnvironment(
    file_name="Reacher_Linux/Reacher.x86_64", no_graphics=False)

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
        agent.critic_target.load_state_dict(
            torch.load('checkpoint_critic.pth'))


def ddpg_train(n_episodes, seed, buffer_size, batch_size, gamma,
               tau, lr_actor, lr_critic, weight_decay):
    scores = []
    scores_deque = deque(maxlen=100)
    agent = Agent(n_agents, state_size, action_size, seed, buffer_size,
                  batch_size, gamma, tau, lr_actor, lr_critic, weight_decay)
    load(agent)
    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[
            brain_name]  # reset the environment
        states = env_info.vector_observations
        agent.reset()  # reset the agent noise
        score = np.zeros(n_agents)
        while True:
            actions = agent.act(states)
            # send the action to the environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards  # update the score
            states = next_states  # roll over the state to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        scores.append(np.mean(score))
        scores_deque.append(np.mean(score))
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(
            i_episode, np.mean(score), np.mean(scores_deque)), end="")
        if n_episodes % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'checkpoint_critic.pth')
        if np.mean(scores_deque) >= 30.0:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(np.arange(len(scores)), scores)
    ax.set(xlabel="Episode #", ylabel="'Score", title="DDPG Network")
    fig.savefig("ddpg_network.pdf")


def trained_agent():
    agent = Agent(n_agents, state_size, action_size, 0, 0, 0, 0, 0, 0, 0, 0)
    load(agent)
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
    parser = argparse.ArgumentParser(
        description='Udacity Deep Reinforcement Learning Nano Degree - Project 2 Continuous Control')
    parser.add_argument('--n_episodes', metavar='', type=int,
                        default=1000, help='maximum number of training episodes')
    parser.add_argument('--seed', metavar='', type=int,
                        default=0, help='seed for stochastic variables')
    parser.add_argument('--buffer_size', metavar='', type=int,
                        default=int(1e5), help='replay buffer size')
    parser.add_argument('--batch_size', metavar='', type=int,
                        default=128, help='minibatch size')
    parser.add_argument('--gamma', metavar='', type=float,
                        default=0.99, help='discount factor')
    parser.add_argument('--tau', metavar='', type=float,
                        default=1e-3, help='for soft update of target parameters')
    parser.add_argument('--lr_actor', metavar='', type=float,
                        default=1e-4, help='learning rate for actor')
    parser.add_argument('--lr_critic', metavar='', type=float,
                        default=1e-4, help='learning rate for agent')
    parser.add_argument('--weight_decay', metavar='', type=int,
                        default=0, help='L2 weight decay')
    parser.add_argument('--train_test', metavar='', type=int,
                        default=0, help='0 to train and 1 to test agent')
    args = parser.parse_args()

    if args.train_test == 0:
        ddpg_train(args.n_episodes, args.seed, args.buffer_size,
                   args.batch_size, args.gamma, args.tau, args.lr_actor,
                   args.lr_critic, args.weight_decay)
    elif args.train_test == 1:
        trained_agent()
    else:
        print("invalid argument for train_test, please use 0 to train and 1 to test agent")
        sys.exit(1)
