# Create Training Dataset

import json
import pandas as pd

with open("data/intents.json") as f:
    intents = json.load(f)

texts = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

print(df.head())
print("Total samples:", len(df))

# Train ML Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X_train = df["text"]
y_train = df["label"]


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X_train, y_train)
print("Model trained on full dataset")


# Save the Trained Model

import joblib

joblib.dump(model, "model/chatbot_model.pkl")
print("Model saved successfully")