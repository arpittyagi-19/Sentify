from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model('model/Mymodel.h5')

# Load the tokenizer
with open('tokenizer/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Sentiment mapping
sentiment_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}

app = Flask(__name__)

# Prediction function
def predict_sentiment(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=50, padding='post')
    prediction = model.predict(padded)
    sentiment = sentiment_map[np.argmax(prediction)]
    return sentiment

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.form['message']
        sentiment = predict_sentiment(message)
        return render_template('index.html', message=message, sentiment=sentiment)
    return render_template('index.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
