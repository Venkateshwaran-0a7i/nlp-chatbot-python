import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stemmer = PorterStemmer()

important_words = {"who", "what", "how", "why", "you"}
stop_words = set(stopwords.words("english"))-important_words

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(tokens)
