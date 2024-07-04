import numpy as np
from tqdm import tqdm

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, roc_auc_score

from loader import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.h2h = nn.Linear(hidden_size, hidden_size)
        # self.h2o = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combination = torch.cat(self.i2h(input) + self.h2h(hidden))
        hidden = self.i2h(combination)
        output = self.i2o(combination)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def train(model, x, y, epochs, lr=0.005, plot_loss=False):
    bar = tqdm(total=epochs)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    nll = nn.NLLLoss()
    all_losses = list()
    for epoch in range(epochs):
        bar.reset()
        cumulative_loss = 0
        opt.zero_grad()

        output, hidden = None, None
        for i in range(len(x)):
            message = x[i]
            hidden = model.init_hidden()
            for j in range(message.size()[0]):
                output, hidden = model(message[j], hidden)

            loss = nll(torch.argmax(output).item(), y)
            cumulative_loss += loss
            loss.backward()
            opt.step()

            bar.set_description(f"Epoch {epoch+1}/{epochs} Loss {cumulative_loss:.4f}")
            bar.update()
        all_losses.append(cumulative_loss)
    if plot_loss:
        plt.figure()
        plt.plot(all_losses)
        plt.show()
        pass
    return model

def predict(input_message):
    print(f'Predicting \"{input_message}\"')
    # with torch.no_grad():
    #     output = evaluate

if __name__ == "__main__":
    data_obj = Dataset("Suicide_Detection.csv")
    n_hidden = 128
    n_categories = 2    # 0 or 1

    x, y = data_obj.data['tensors'], data_obj.data['class']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, random_state = 12)
    xtrain, xtest, ytrain, ytest = torch.Tensor(xtrain).to(device), torch.Tensor(xtest).to(device), torch.Tensor(ytrain).to(device), torch.Tensor(ytest).to(device)
    model = RNN(n_chars, n_hidden, n_categories).to(device)
    model = train(model, xtrain, ytrain, epochs=500)

    from pathlib import Path
    Path("../models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), '../models/torch_RNN.pth')
