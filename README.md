🤖 NLP Chatbot using Python & Machine Learning

An intelligent intent-based chatbot built using Python, NLP, and Machine Learning.
The chatbot classifies user input into intents using TF-IDF + Logistic Regression and responds with confidence-based logic.

This project demonstrates real-world NLP pipeline design, model training, inference, and clean project structuring.

🚀 Features

Intent classification using Machine Learning

TF-IDF text vectorization

Logistic Regression classifier

Confidence-based fallback handling

Modular, industry-style Python project structure

Trained model persistence (.pkl)

Easy to extend with new intents

🧠 How It Works (Pipeline)
User Input
   ↓
TF-IDF Vectorization
   ↓
ML Intent Classifier (Logistic Regression)
   ↓
Confidence Check
   ↓
Response OR Fallback

🛠 Tech Stack

Python

NLTK – text preprocessing

Scikit-learn – TF-IDF & ML model

Joblib – model persistence

Git & GitHub – version control

📁 Project Structure
AI/chatbot/
├── data/
│   └── intents.json
├── model/
│   └── chatbot_model.pkl
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── preprocess.py
│   └── train_model.py
├── .gitignore
└── main.py

⚙️ Setup & Installation
1️⃣ Clone the repository
git clone https://github.com/Venkateshwaran-0a7i/nlp-chatbot-python.git
cd nlp-chatbot-python/AI/chatbot

2️⃣ Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Download NLTK resources
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

🧪 Train the Model
python src/train_model.py


This will:

Train the intent classifier

Save the model to model/chatbot_model.pkl

▶️ Run the Chatbot
python main.py

Example
You: hi
Bot: Hello!

You: who are you
Bot: I am a Python NLP-based chatbot.

🎯 Key Learning Outcomes

Designed an NLP pipeline from scratch

Built a supervised ML intent classifier

Handled small-dataset ML challenges

Implemented confidence-based inference

Managed clean Git version control

📌 Future Improvements

Data augmentation for higher accuracy

Lemmatization instead of stemming

REST API using FastAPI

Web-based chat interface

Embedding-based semantic search