import pickle
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
root_path = "C:/Users/khiemlh/Desktop/DraftSpace/Sentiment Analysis/"

train_df = pd.read_csv(root_path + "data/train.csv")
eval_df = pd.read_csv(root_path + "data/eval.csv")

train_cmts = train_df["comment"].values.astype("U")
eval_cmts = eval_df["comment"].values.astype("U")

train_labels = train_df["label"].values
eval_labels = eval_df["label"].values

tokenizer=ViTokenizer.tokenize
with open("stopwords.txt", encoding="utf-8") as f:
    stopwords = [line.rstrip() for line in f]

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=list(stopwords), ngram_range=(1, 3), max_features=5000).fit(train_cmts)
train_tfidf = tfidf_vectorizer.transform(train_cmts)
eval_tfidf = tfidf_vectorizer.transform(eval_cmts)
pickle.dump(tfidf_vectorizer, open(root_path + "ckps/TfidfVectorizer.pkl", "wb"))

model = SVC(C=2.0).fit(train_tfidf, train_labels)
train_preds = model.predict(train_tfidf)
eval_preds = model.predict(eval_tfidf)
pickle.dump(model, open(root_path + "ckps/SVC.pkl", "wb"))

print("Train Accuracy: {:.4f}".format(accuracy_score(train_labels, train_preds)))
print("Eval Accuracy: {:.4f}".format(accuracy_score(eval_labels, eval_preds)))