import pandas as pd

import pickle

import re
import emoji
import string
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


df = pd.read_excel("cyberbullying_instagram.xls")

# Encode Label
df['Label'] = df['Kategori'].apply(lambda x: 1 if x == "Bullying" else 0)

    
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):

    # 1. Casefolding
    text = text.lower()

    # 2. Hapus HTML tag
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    # 3. Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 4. Hapus Emoji
    text = emoji.demojize(text)

    # 5. Hapus emoji & karakter non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 6. Hapus angka
    text = re.sub(r'\d+', '', text)

    # 7. Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 8. Huruskan whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Tokenisasi
    tokens = word_tokenize(text)

    # 10. Hapus stopword
    tokens = [t for t in tokens if t not in stop_words]

    # Gabung lagi jadi teks
    return " ".join(tokens)

df['Comment'] = df['Komentar'].apply(clean_text)


tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df['Comment'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred) 
nb_precision = precision_score(y_test, nb_pred) 
nb_recall = recall_score(y_test, nb_pred) 


filename = "model.pkl"

pickle.dump(nb_model, open(filename, 'wb'))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))