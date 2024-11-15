import os
import sys
sys.stdout = sys.stderr  # Directing stdout to stderr to avoid WinError 6

from flask import Flask, render_template, request
import pickle
import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords

# Download the stopwords list if you haven't already
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Define the text_preprocessor function
def text_preprocessor(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Define TextPreprocessor class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [text_preprocessor(text) for text in X]

# Load the saved pipeline
with open('spam_ham_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the form
        text = request.form['text']

        # Make the prediction
        prediction = pipeline.predict([text])[0]

        # Display result with color based on prediction
        if prediction == 1:
            result = "Spam"
            color = "red"
        else:
            result = "Ham"
            color = "green"

        return render_template('index.html', prediction=result, color=color)

if __name__ == '__main__':
    app.run(debug=True)



