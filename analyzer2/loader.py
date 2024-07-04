import torch
import numpy as np
import pandas as pd
import json
import string
import emoji
from nltk.tokenize import word_tokenize, wordpunct_tokenize
# from keras_preprocessing.text import Tokenizer

all_chars = string.ascii_letters + string.punctuation + string.digits
n_chars = len(all_chars)

class Message(str):
    def __init__(self, message):
        super(Message, self).__init__()
        self.message = message

    def extract_emojis(self):
        return Message(emoji.demojize(self.message, delimiters=("", "")))

    def char_embedding(self):
        """Embedding of each letter using a tensor/array as a one-hot vector"""
        tensor = torch.zeros(len(self.message), 1, n_chars)
        for num, letter in enumerate(self.message):
            ind = all_chars.find(letter)
            if ind != -1:
                tensor[num][0][ind] = 1
        return tensor

    def embed(self):
        self.message = self.extract_emojis()
        self.message = self.char_embedding()
        return self.message

    def word_embedding(self):
        pass

    def ngram(self):
        context_size = 2
        embedding_dim = 10
        message = self.message.split()
        vocab = set(message)
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        ngrams = [([message[i-j-1] for j in range(context_size)], message[i]) for i in range(context_size, len(message))]
        pass

class Dataset():
    def __init__(self, filename: str, filetype: str = "csv"):
        self.data = None
        if filetype == "csv":
            self.data = pd.read_csv(f"../data/{filename}")
            self.data.drop(["Unnamed: 0"], axis=1, inplace=True)
            self.data[["class"]] = (self.data[["class"]] == "suicide").astype("int16")
        else: #filetype == "json":
            with open(f"../data/{filename}") as f:
                self.data = json.load(f)
        self.data.insert(1, "tensors", [Message(self.data.iloc[i, 0]).embed() for i in range(self.data.shape[0])], allow_duplicates=True)
        print(self.data.head())

    def print_data(self):
        if self.data is not None:
            print(self.data.head())
            return self.data
        print("No data valid given")

if __name__ == "__main__":
    data_obj = Dataset("Suicide_Detection.csv")
    for i in range(5):
        message = Message(data_obj.data.iloc[i, 0]).embed()
        print(message.shape)
