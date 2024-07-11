import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sys import argv
from pathlib import Path
Path("../models").mkdir(parents=True, exist_ok=True)

import torch; torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
generator = torch.Generator().manual_seed(0)
import torch.nn as nn

from loader import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combination = torch.cat((input, hidden), 1)
        hidden = self.i2h(combination)
        output = self.i2o(combination)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).to(device)

def get_category(output):
    return torch.argmax(output).item()

def training(model, xtrain, ytrain, iterations):
    """Full backwards function"""
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    lr = 0.00005
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def train(model, x, y):
        """Forward step"""
        hidden = model.init_hidden()

        output = 0
        for i in range(len(x)):
            output, hidden = model(x[i], hidden)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return model, output, loss.item()

    current_loss = 0
    all_losses = list()
    plot_steps = 1000
    n_iters = iterations
    loss = 0
    bar = tqdm(total=n_iters)
    for i in range(n_iters):
        x, y = data_obj.embed_message(xtrain[i]).to(device), torch.tensor([ytrain[i]]).to(device)

        model, output, loss = train(model, x, y)

        current_loss += loss
        bar.set_description(f"Iteration {i+1}/{n_iters} Loss {current_loss:.2f}")
        bar.update()

        if (i+1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()
    return model

if __name__ == "__main__":
    """Training/Testing (on Suicide_Detection.csv)"""

    """To train a word embedding model"""
    # data_obj = Dataset(argv[1])
    # data_obj = Dataset("Suicide_Detection.csv")

    word_vectors = gensim.models.Word2Vec.load('preprocessing/word_vectors')
    data_obj = Dataset("Suicide_Detection.csv", model=word_vectors)

    n_dims = data_obj.tensor_shape
    n_hidden = 128
    n_categories = 2    # 0 or 1

    """Training RNN"""
    model = RNN(n_dims, n_hidden, n_categories).to(device)
    xtrain, ytrain = list(data_obj.data["text"]), list(data_obj.data["class"])
    # iterations = int(argv[2])
    iterations = 100000
    model_name = '../models/torch_RNN2.pth'
    model = training(model, xtrain, ytrain, iterations)
    torch.save(model.state_dict(), model_name)

    """Testing"""
    rnn = RNN(n_dims, n_hidden, n_categories).to(device)
    rnn.load_state_dict(torch.load(model_name))
    rnn.eval()
    tpr, tnr, fpr, fnr = 0, 0, 0, 0
    tn, fp, fn, tp = 0, 0, 0, 0
    # n_iters = int(argv[3])
    n_iters = 15000
    bar = tqdm(total=n_iters)
    for num in range(n_iters):
        x, y = data_obj.embed_message(xtrain[num+iterations+1]).to(device), torch.tensor([ytrain[num+iterations+1]]).to(device)
        hidden = rnn.init_hidden()

        output = 0
        for i in range(len(x)):
            output, hidden = rnn(x[i], hidden)
        output = get_category(output)
        y = int(y)

        if output and y:
            tp += 1
        elif output and not y:
            fn += 1
        elif not output and not y:
            tn += 1
        elif not output and y:
            fp += 1

        try:
            acc = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            acc = 0

        bar.set_description(f"Iter {num+1}/{n_iters} ACC: {acc:.4f}")
        bar.update()


    calculate_metrics(tp, tn, fp, fn, print_res=True)
