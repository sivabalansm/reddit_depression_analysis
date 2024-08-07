#! /bin/python3
# Code autogenerated from Jupyter notebook with a script. May contain errors.
import numpy as np
import matplotlib.pyplot as plt

import pickle

def load_set(directory):
    try:
        with open(f"{directory}/texts2.pkl", "rb") as fp:
            processed_texts = pickle.load(fp)
        
        with open(f"{directory}/labels2.pkl", "rb") as fp:
            labels = pickle.load(fp)
    
    except:
        print(f'{directory} files not found. Please run the preprocess.ipynb before!')
    
    return processed_texts, labels

processed_texts, labels = load_set('train')
processed_val_texts, val_labels = load_set('val')
processed_test_texts, test_labels = load_set('test')

from datasets import Dataset

train_ds = Dataset.from_dict({ 'text': processed_texts, 'label': labels })
val_ds = Dataset.from_dict({ 'text': processed_val_texts, 'label': val_labels })

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

tokenized_texts = train_ds.map(preprocess_function, batched=True)
tokenized_val_texts = val_ds.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "Depressed", 1: "Happy"}
label2id = {"Depressed": 0, "Happy": 1}

from transformers import create_optimizer

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_texts) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

tf_train_set = model.prepare_tf_dataset(
    tokenized_texts,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_val_set = model.prepare_tf_dataset(
    tokenized_val_texts,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

import tensorflow as tf

model.compile(optimizer=optimizer)

from transformers.keras_callbacks import KerasMetricCallback

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_val_set)

callbacks = [metric_callback]

model.fit(x=tf_train_set, validation_data=tf_val_set, epochs=num_epochs, callbacks=callbacks)

model.summary()

test_ds = Dataset.from_dict({ 'text': processed_test_texts, 'label': test_labels })
tokenized_test_texts = test_ds.map(preprocess_function, batched=True)

tokenized_test_texts = [tokenizer(text, truncation=True, return_tensors='tf') for text in processed_test_texts]

logits = [model(**tokenized_test_text).logits for tokenized_test_text in tokenized_test_texts]

y_pred = [int(tf.math.argmax(logit, axis=-1)[0]) for logit in logits]
# y_pred = (y_pred > 0.5).astype(int)

# y_pred = np.array(list(map(lambda x: x[0], y_pred)))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()

fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

fnr = fn / (fn + tp)
print(f"False Negative Rate: {fnr:.4f}")

print(f'accuracy_score {accuracy_score(test_labels, y_pred):.3f}')
print(f'precision_score {precision_score(test_labels, y_pred):.3f}')
print(f'recall_score {recall_score(test_labels, y_pred):.3f}')
print(f'f1_score {f1_score(test_labels, y_pred):.3f}')

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(test_labels, y_pred, normalize="true",
                                        values_format=".0%")
import pickle

with open('./models/hug_clf2.pkl', 'wb') as fp:
  pickle.dump(model, fp)

with open('./models/hug_tok2.pkl', 'wb') as fp:
  pickle.dump(tokenizer, fp)

plt.show()
