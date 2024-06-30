import torch
import keras
import numpy as np
import pandas as pd
import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras_preprocessing.text import Tokenizer
import string
import emoji

nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words('english'))


# can also use demoji package
# demoji.download_codes()
all_letters = string.ascii_letters

class Message():
    def __init__(self, message: str):
        self.text = message

    def extract_emojis(self, message):
        return emoji.demojize(message, delimiters=("", ""))

    def tokenize(self):
        pass

    # def messageToTensor(self, message):
    #     tensor = torch.zeros(len(message), 1, n_letters)
    #     message = message.split()

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
    # Test emoji replacement
    message = Message("hello how are you 😜")
    message = message.extract_emojis(message.text)
    print(message)
