import numpy as np
from tqdm import tqdm

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.softmax = nn.LogSoftmax(dim=1)

        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()

        # Loss

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output
