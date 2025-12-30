# import json
# import random
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from src.preprocess import preprocess

# fallback_responses = [
#     "I'm not sure I understood that. Could you rephrase your question?",
#     "Sorry, I didn't get that. Try asking in a different way.",
#     "I’m still learning 😊. You can ask me what I can do or type 'help'.",
#     "That’s outside my current knowledge. Try a simpler question."
# ]


# with open("data/intents.json") as f:
#     intents = json.load(f)

# corpus = []
# responses = []

# for intent in intents["intents"]:
#     for pattern in intent["patterns"]:
#         corpus.append(preprocess(pattern))
#         responses.append(intent["responses"])

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)

# def get_response(user_input):
#     user_input = preprocess(user_input)
#     user_vec = vectorizer.transform([user_input])
#     similarity = cosine_similarity(user_vec, X)
#     idx = similarity.argmax()
    
#     if similarity[0][idx] > 0.15:
#         return random.choice(responses[idx])
#     else:
#         return random.choice(fallback_responses)

import joblib
import json
import random
from src.train_model import model

model = joblib.load("model/chatbot_model.pkl")

with open("data/intents.json") as f:
    intents = json.load(f)

responses_map = {
    intent["tag"]: intent["responses"]
    for intent in intents["intents"]
}

fallback_responses = [
    "I'm not sure I understood that. Try rephrasing.",
    "I didn't get that. You can type 'help' to see what I can do."
]

def get_response(user_input):
    probs = model.predict_proba([user_input])[0]
    max_prob = probs.max()
    intent = model.predict([user_input])[0]

    if max_prob >= 0.25:
        return random.choice(responses_map[intent])
    else:
        return random.choice(fallback_responses)

