# Import necessary libraries
from sklearn import svm
import numpy as np
from tqdm import tqdm
from loader import *

def run_model(kernel: str, x, y):
    # Fit the SVM model
    linear = svm.SVC(kernel=kernel)
    linear.fit(x, y)

    # Predict using the SVM model
    predictions = linear.predict(x)

    # Evaluate the predictions
    accuracy = linear.score(x, y)
    print(f"Accuracy of {kernel} SVM:", accuracy)

word_vectors = gensim.models.Word2Vec.load('../models/word_vectors')
data_obj = Dataset("Suicide_Detection.csv", model=word_vectors)

x = list()
print("Loading tensors...")
data = data_obj.data["text"]
bar = tqdm(total=len(data))
for i in range(len(data)):
    x.append(data_obj.embed_message_mean(data[i], word_vectors))
    bar.set_description(f"Iteration {i+1}/{len(data)}")
    bar.update()
x = np.array(x)
y = np.array(data_obj.data["class"])

run_model('linear', x, y)
run_model('poly', x, y)
run_model('rbf', x, y)
run_model('sigmoid', x, y)
