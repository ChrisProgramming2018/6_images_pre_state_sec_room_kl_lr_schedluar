import torch
import torch.nn as nn
import torch.nn.functional as F



class RNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_dim, seed, fc1_units=64*2, fc2_units=64*2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(RNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)




class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=64,fc2_units=64, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DClassifier(nn.Module):
    """ Classifier Model."""

    def __init__(self, state_size, action_dim, seed, fc1_units=64,fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Classifier, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

    def forward(self, state, train=False, stats=False):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.dropout(x,training=train)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(x,training=train)
        output = self.fc3(x)
        return output



class DQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RvNetwork(nn.Module):
    """ Classifier Model."""

    def __init__(self, state_size, action_dim, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(RNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

    def forward(self, state, train=False, stats=False):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.dropout(x,training=train)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(x,training=train)
        output = self.fc3(x)
        return output




class Encoder(nn.Module):
    def __init__(self, config, D_out=200,conv_channels=[16, 32], kernel_sizes=[8, 4], strides=[4,2]):
        super(Encoder, self).__init__()
        # Defining the first Critic neural network
        channels = config["history_length"]
        self.conv_1 =  torch.nn.Conv2d(channels, conv_channels[0], kernel_sizes[0], strides[0])
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 =  torch.nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1], strides[1])
        self.relu_2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.Linear  = torch.nn.Linear(2592, D_out)
        self.relu_3 = torch.nn.ReLU()
    def create_vector(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim: # case of RNN input
            obs = obs.view(-1, *obs_shape[2:])
        x = self.conv_1(obs)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.flatten(x)
        obs = self.relu_3(self.Linear(x))
        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs                       

 
class Classifier(nn.Module):
    """ Classifier Model."""
    def __init__(self, state_size, action_dim, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
          ======
              state_size (int): Dimension of each state
              action_size (int): Dimension of each action
              seed (int): Random seed
             fc1_units (int): Number of nodes in first hidden layer
              fc2_units (int): Number of nodes in second hidden layer
        """
        super(Classifier, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
