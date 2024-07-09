import numpy as np
import pandas as pd
import json, string, emoji
import torch, nltk
from nltk.tokenize import sent_tokenize, word_tokenize #, wordpunct_tokenize
import gensim
from gensim.models import KeyedVectors
from pathlib import Path

# from keras_preprocessing.text import Tokenizer

all_chars = string.ascii_letters + string.punctuation + string.digits
n_chars = len(all_chars)

class Message(str):

    """For character-level embedding (computation time too slow for long messages)"""
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

class Dataset():
    def __init__(self, filename: str, filetype: str = "csv", model: gensim.models.word2vec.Word2Vec | None = None):
        print(f"Loading {filename} dataset...")
        self.data = None
        self.training_data = list()
        if filetype == "csv":
            self.data = pd.read_csv(f"../data/{filename}")
            self.data.drop(["Unnamed: 0"], axis=1, inplace=True)
            self.data[["class"]] = (self.data[["class"]] == "suicide").astype("int16")
        else: #filetype == "json":
            with open(f"../data/{filename}") as f:
                self.data = json.load(f)
        if model is None:
            model = self.train_embedding_model(save_model=False)
        self.tensor_shape = model.wv.get_vector(0).shape[0]

    def __str__(self):
        if self.data is not None:
            return str(self.data)
        return "No data valid given"

    def embed_message(self, message: str, word_vectors: gensim.models.word2vec.Word2Vec):
        """Embedding for RNN"""
        tokenized = word_tokenize(message)
        embedded = torch.zeros(len(tokenized), 1, self.tensor_shape)
        for idx, word in enumerate(tokenized):
            try:
                word_tensor = torch.tensor(word_vectors.wv.get_vector(word))
                embedded[idx][0] = word_tensor
            except KeyError:
                pass
        return embedded

    def embed_message_mean(self, message: str, word_vectors: gensim.models.word2vec.Word2Vec):
        """Embedding for SVM"""
        tokenized = word_tokenize(message)
        embedded = list()
        for word in tokenized:
            embedded.append(word)
        embedded = np.array(embedded)
        mean = word_vectors.wv.get_mean_vector(embedded)
        return mean

    def train_embedding_model(self, save_model=True):
        """Training the embedding model (get a vector for every word)"""
        nltk.download('popular')
        for i in range(self.data.shape[0]):
            text = self.data.iloc[i, 0]
            text.replace("\n", " ")
            for j in sent_tokenize(text):
                temp = list()
                for k in word_tokenize(j):
                    k = emoji.demojize(k, delimiters=("", ""))
                    temp.append(k.lower())
                self.training_data.append(temp)
        word_model = gensim.models.Word2Vec(self.training_data, min_count=1)

        if save_model:
            Path("../models").mkdir(parents=True, exist_ok=True)
            word_model.save('../models/word_vectors')
        return word_model

if __name__ == "__main__":
    """Testing"""
    # word_vectors = data_obj.train_embedding_model()
    word_vectors = gensim.models.Word2Vec.load('../models/word_vectors')
    data_obj = Dataset("Suicide_Detection.csv", model=word_vectors)
    print(data_obj)
    mean = data_obj.embed_message_mean(data_obj.data.iloc[0, 0], word_vectors)
    print(mean)
