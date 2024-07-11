import numpy as np
import pandas as pd
import json, sys
import nltk
from nltk.tokenize import word_tokenize
import gensim
from alive_progress import alive_it
from pathlib import Path
import pickle
import spacy
nlp = spacy.load('en_core_web_trf', disable=['parser', 'ner', 'transformer'])

class Dataset():
    def __init__(self, filename:str, lemma_dir: str | None = None, filetype: str = "csv", model: gensim.models.word2vec.Word2Vec | None = None):
        print(f"Loading {filename} dataset...")
        if filetype == "csv":
            self.data = pd.read_csv(f"../data/{filename}")
            self.data.drop(["Unnamed: 0"], axis=1, inplace=True)
            self.data[["class"]] = (self.data[["class"]] == "suicide").astype("int16")
            if lemma_dir is None:
                self.processed_texts, self.labels = preprocess_set(self.data)
            else:
                self.processed_texts, self.labels = load_set(lemma_dir)
        else:   #filetype == "json":
            with open(f"../data/{filename}") as f:
                self.data = json.load(f)
        if model is None:
            model = self.train_embedding_model()
        self.word_model = model
        self.tensor_shape = self.word_model.wv.get_vector(0).shape[0]

    def __str__(self):
        if self.data is not None:
            return str(self.data)
        return "No data valid given"

    def embed_message_mean(self):
        """Embedding for SVM"""
        word_vectors = self.word_model
        mean = list()
        for message in alive_it(range(len(self.processed_texts)), total=len(self.processed_texts)):
            tokenized = word_tokenize(self.processed_texts[message])
            try:
                mean.append(word_vectors.wv.get_mean_vector(tokenized))
            except ValueError:
                mean.append(word_vectors.wv.get_mean_vector(self.data.iloc[message, 0]))
        return mean

    def train_embedding_model(self, directory="preprocessing", save_model=True):
        """Training the embedding model (get a vector for every word)"""
        print("Loading word embedding model...")
        nltk.download('popular')
        self.training_data = list()
        for text in alive_it(self.processed_texts, total=len(self.processed_texts)):
            temp = list()
            for i in word_tokenize(text):
                temp.append(i.lower())
            self.training_data.append(temp)
        word_model = gensim.models.Word2Vec(self.training_data, min_count=1)

        if save_model:
            Path(directory).mkdir(parents=True, exist_ok=True)
            word_model.save(f'{directory}/word_vectors.wv')
        self.word_model = word_model
        return word_model

def preprocess_set(data, directory="preprocessing", save_file=False):
    print(f'Preprocessing {directory} data')

    texts = data.copy()['text']
    labels = data.copy()['class']
    texts = [' '.join(text.split()) for text in texts]

    docs = (doc for doc in (nlp.pipe(texts)))
    processed_texts = []
    for doc in alive_it(docs, total=len(texts)):
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        processed_texts.append(' '.join(lemmas))

    labels = np.array(labels)

    if save_file:
        Path(f"{directory}").mkdir(parents=True, exist_ok=True)

        with open(f"{directory}/lemma_texts.pkl", "wb") as fp:
            pickle.dump(processed_texts, fp)

        with open(f"{directory}/lemma_labels.pkl", "wb") as fp:
            pickle.dump(labels, fp)
    return processed_texts, labels

def load_set(directory="preprocessing"):
    processed_texts, labels = list(), list()
    try:
        with open(f"{directory}/lemma_texts.pkl", "rb") as fp:
            processed_texts = pickle.load(fp)

        with open(f"{directory}/lemma_labels.pkl", "rb") as fp:
            labels = pickle.load(fp)

    except FileNotFoundError:
        print(f'{directory} files not found.')

    labels = np.array(labels)

    return processed_texts, labels

def calculate_metrics(tp, tn, fp, fn, print_res=True):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)

    if print_res:
        print(f"True Positive Rate: {tpr:.4f}")
        print(f"True Positive Rate: {tnr:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return tpr, tnr, fpr, fnr, acc, prec, rec, f1


if __name__ == "__main__":
    """Testing"""
    data_obj = Dataset(filename=sys.argv[1])
    # word_vectors = gensim.models.Word2Vec.load('preprocessing/word_vectors.wv')
