import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import warnings
warnings.filterwarnings('ignore')
# Load the IMDB dataset (replace with your dataset path)
data = pd.read_csv("imdb_dataset.csv")

# Preprocessing
def preprocess_text(text):
  # Tokenization
  tokens = nltk.word_tokenize(text.lower())
  
  # Remove stop words
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]
  
  # Stemming
  stemmer = PorterStemmer()
  tokens = [stemmer.stem(word) for word in tokens]
  
  return ' '.join(tokens)

data['review'] = data['review'].apply(preprocess_text)
X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a pipeline for text preprocessing and classification
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model
text_clf.fit(X_train, y_train)
# Make predictions on the test set
predicted = text_clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predicted))
print("Confusion Matrix:\n", confusion_matrix(y_test, predicted))
print("Classification Report:\n", classification_report(y_test, predicted))
