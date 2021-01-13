import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, action_size, capacity, batch_size, image_pad, device):
        self.action_size = action_size
        self.capacity = capacity
        self.device = device
        self.batch_size = batch_size
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.int8)
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.k = 0
        self.kl_threshold = 0.1

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self.k +=1
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, qnetwork_local, encoder):
        idxs = []
        final_idxs = []
        # get idx only for kl bigger than 1
        if len(self.possible_idx) < 200:
            print("set")
            self.possible_idx = self.save_pos_idx
            self.kl_threshold -= 0.01
        batch_size = self.batch_size
        while True:
            idxs = np.random.choice(self.possible_idx, batch_size, replace=False)
            for i in idxs:
                states = torch.as_tensor(self.obses[i], device=self.device)
                actions = torch.as_tensor(self.actions[i], device=self.device)
                states = torch.as_tensor(states, device=self.device).unsqueeze(0)
                states = states.type(torch.float32)
                states = encoder.create_vector(states.detach())
                one_hot = torch.Tensor([0 for i in range(self.action_size)], device="cpu")
                one_hot[actions.item()] = 1
                with torch.no_grad():
                    q_values = qnetwork_local(states.detach()).detach()
                    soft_q = F.softmax(q_values, dim=1).to("cpu")
                    kl_q =  F.kl_div(soft_q.log(), one_hot, None, None, 'sum')
                    if kl_q >= self.kl_threshold:
                        final_idxs.append(i)
                    else:
                        self.possible_idx.remove(i)

            batch_size = self.batch_size - len(final_idxs)
            if batch_size <= 0:
                break
        idxs = final_idxs
        print(len(self.possible_idx))
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses = torch.as_tensor(obses, device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        return obses, next_obses, actions


    def expert_policy(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        obses = obses.type(torch.float32)
        next_obses = next_obses.type(torch.float32)
 
        #obses = self.aug_trans(obses)
        #next_obses = self.aug_trans(next_obses)
        
        return obses, next_obses, actions




    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
    

        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)
        
        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)
        
        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)
        
        with open(filename + '/not_dones.npy', 'wb') as f:
            np.save(f, self.not_dones)
        
        with open(filename + '/not_dones_no_max.npy', 'wb') as f:
            np.save(f, self.not_dones_no_max)

        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))

        print("save buffer to {}".format(filename))
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)
        
        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)
        
        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())
        self.possible_idx = [i for i in range(self.idx)]
        self.save_pos_idx = copy.deepcopy(self.possible_idx)
