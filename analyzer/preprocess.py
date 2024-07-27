#! /bin/python3
# Code autogenerated from Jupyter notebook with a script. May contain errors.
import numpy as np
import pandas as pd

posts = pd.read_csv("../data/Suicide_Detection.csv")
posts.drop(["Unnamed: 0"], axis=1, inplace=True)
posts[["class"]] = (posts[["class"]] == "suicide").astype("int16")

posts.head()

posts.describe()

# Only select posts with a word count of between 20 and 1000 words
# posts = posts[np.array([20 < len(post.split()) < 1000 for post in posts["text"]])]

post_lengths = [len(post.split()) for post in posts["text"]]

import matplotlib.pyplot as plt

plt.hist(post_lengths, bins=100)
# plt.show()

from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(posts, test_size=0.1, random_state=42)

strat_train_set, strat_val_set = train_test_split(posts, test_size=1/9, random_state=1)

strat_train_set.head()

import spacy

print(spacy.prefer_gpu())

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

nlp.pipe_names

doc = nlp('I was reading the paper.')
print([token.text for token in doc if not token.is_stop and not token.is_punct])

import pickle
from tqdm import tqdm

def preprocess_set(set, directory):
    print(f'Preprocessing {directory} data')

    texts = set.copy()['text']
    labels = set.copy()['class']
    texts = [' '.join(text.split()[:500]) for text in texts]

    docs = (doc for doc in (nlp.pipe(texts)))
    processed_texts = []
    for doc in tqdm(docs, total=len(texts), ncols=80):
        lemmas = [token.text for token in doc if not token.is_stop and not token.is_punct]
        # lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        processed_texts.append(' '.join(lemmas))
    
    labels = np.array(labels)

    from pathlib import Path
    Path(f"{directory}").mkdir(parents=True, exist_ok=True)

    with open(f"{directory}/texts2.pkl", "wb") as fp:
        pickle.dump(processed_texts, fp)
    
    with open(f"{directory}/labels2.pkl", "wb") as fp:
        pickle.dump(labels, fp)

    return processed_texts, labels

len(strat_train_set)

preprocess_set(strat_train_set, 'train')
preprocess_set(strat_val_set, 'val')
preprocess_set(strat_test_set, 'test')

print()

