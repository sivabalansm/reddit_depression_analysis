from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import svm
from sys import argv
from loader import *

def run_model(kernel: str, x, y, degree: int = 3):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.14, random_state=42)
    # Fit the SVM model
    model = None
    if kernel == "poly":
        model = svm.SVC(kernel=kernel, degree=degree)
    else:
        model = svm.SVC(kernel=kernel, degree=degree)
    model.fit(xtrain, ytrain)

    # Predict using the SVM model
    print("Predicting...")
    predictions = model.predict(xtest)

    # Evaluate the predictions
    print("Calculating metrics...")
    tn, fp, fn, tp = confusion_matrix(ytest, predictions).ravel()
    ConfusionMatrixDisplay.from_predictions(ytest, predictions, normalize="true", values_format=".0%")
    plt.savefig(f"results/{kernel}_{degree}_SVM.png")
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    acc = accuracy_score(ytest, predictions)
    rec = recall_score(ytest, predictions)
    prec = precision_score(ytest, predictions)
    f1 = f1_score(ytest, predictions)

    with open("results/results_svm.txt", "a+") as f:
        print(f"{kernel.capitalize()}, degree = {degree} SVM", file=f)
        print(f"Confusion Matrix of {kernel} SVM: {tp}, {tn}, {fp}, {fn}", file=f)
        print(f"True Positive Rate of {kernel} SVM: {tpr:.4f}", file=f)
        print(f"True Negative Rate of {kernel} SVM: {tnr:.4f}", file=f)
        print(f"False Positive Rate of {kernel} SVM: {fpr:.4f}", file=f)
        print(f"False Negative Rate of {kernel} SVM: {fnr:.4f}", file=f)
        print(f"Accuracy of {kernel} SVM: {acc:.4f}", file=f)
        print(f"Precision of {kernel} SVM: {prec:.4f}", file=f)
        print(f"Recall of {kernel} SVM: {rec:.4f}", file=f)
        print(f"F1 Score of {kernel} SVM: {f1:.4f} \n", file=f)

word_vectors = gensim.models.Word2Vec.load('preprocessing/word_vectors.wv')
data_obj = Dataset(filename="Suicide_Detection.csv", lemma_dir="preprocessing", model=word_vectors)

print("Loading tensors...")
# x = list()
# data = data_obj.data["text"]
# bar = tqdm(total=len(data))
# for i in range(len(data)):
#     x.append(data_obj.embed_message_mean(data[i]))
#     bar.set_description(f"Iteration {i+1}/{len(data)}")
#     bar.update()
x = data_obj.embed_message_mean()
y = data_obj.labels

# run_model('linear', x, y)
run_model('poly', x, y, degree=4)
# run_model('rbf', x, y)
# run_model('sigmoid', x, y)
