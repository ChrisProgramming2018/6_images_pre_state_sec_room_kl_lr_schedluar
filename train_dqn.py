import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent_iql import Agent, mkdir
import json
import argparse
from helper import FrameStack
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

def main(args):
    scores = dqn(args)



def dqn(args, n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    with open ("param.json", "r") as f:
        config = json.load(f)
    
    agent = DQNAgent(state_size=200, action_size=7,  config=config)
    agent_r = Agent(state_size=200, action_size=7,  config=config)
    agent_r.load("models-28_11_2020_12:54:29/3800-")
    env = gym.make('MontezumaRevenge-v0')
    env  = FrameStack(env, args)
    env.seed(0)
    memory =  ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        obs_env = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(obs_env, eps)
            action_tensor = torch.from_numpy(np.array(action)).float().unsqueeze(0).to(agent_r.device)
            obs = torch.from_numpy(obs_env).float().unsqueeze(0).to(agent_r.device)
            action_tensor = action_tensor.type(torch.int64)
            state = agent_r.encoder.create_vector(obs.div_(255))
            reward = agent_r.R_local(state).gather(1, action_tensor.unsqueeze(0))
            next_obs, _ , done, _ = env.step(action)
            memory.add(obs_env, action, reward.item(), next_obs, done, done)
            agent.step(memory)
            obs_env = next_obs
            score += reward.item()
            if done:
                print("")
                print("")
                print("")
                print("Model reward ", score)
                agent_r.save("Q_model/-{}".format(i_episode))
                break 
        scores_window.append(score)       # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint_q-{}.pth'.format(i_episode))
            torch.save(agent.encoder.state_dict(), 'models/checkpoint_e-{}.pth'.format(i_episode))
    return scores
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="MontezumaRevenge-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--buffer_size', default=20000, type=int)
    arg = parser.parse_args()
    main(arg)
