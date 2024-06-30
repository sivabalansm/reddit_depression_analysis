import torch
import numpy as np
import pandas as pd
import json
import string
import emoji
from nltk.tokenize import word_tokenize, wordpunct_tokenize
# from keras_preprocessing.text import Tokenizer


# can also use demoji package
# demoji.download_codes()

class Message():
    def __init__(self, message):
        self.message = message
        self.all_chars = string.ascii_letters
        self.n_chars = len(self.all_chars)

    def extract_emojis(self):
        return emoji.demojize(self.message, delimiters=("", ""))

    def word_embedding(self):
        pass

    def ngram(self):
        context_size = 2
        embedding_dim = 10
        message = self.message.split()
        vocab = set(message)
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        ngrams = [([message[i-j-1] for j in range(context_size)], message[i]) for i in range(context_size, len(message))]

    def char_embedding(self):
        """Embedding of each letter using a tensor/array as a one-hot vector"""
        tensor = torch.zeros(len(self.message), 1, self.n_chars)
        for num, letter in enumerate(self.message):
            tensor[num][0][self.all_chars.find(letter)] = 1
        return tensor

class Dataset():
    def __init__(self, filename: str, filetype: str):
        self.data = None
        if filetype == "json":
            with open(f"../data/{filename}") as f:
                self.data = json.load(f)
        elif filetype == "csv":
            posts = pd.read_csv(f"../data/{filename}")
            # posts.drop(["Unnamed: 0"], axis=1, inplace=True)
            # posts[["class"]] = (posts[["class"]] == "suicide").astype("int16")

    def print_data(self):
        pass


if __name__ == "__main__":
    message = Message("hello how are you 😜").extract_emojis()
    print(message)
