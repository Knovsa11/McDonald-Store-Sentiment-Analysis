import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow_hub import KerasLayer
import nltk

# Download resource stopwords jika belum tersedia
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize

# Memuat model dengan menentukan custom_objects
model = load_model('sentiment_model.h5', custom_objects={'KerasLayer': KerasLayer})

# Membuat stopword
stpwds_en = set(stopwords.words('english'))

# Membuat variabel stopword tambahan
additional_stopwords = ['food','order', 'McDonald', '½', 'ï', '½ï', 'drive thru']

lemmatizer = WordNetLemmatizer()

# Membuat fungsi untuk text preprocessing
def text_preprocessing(text):
    # Case folding
    text = text.lower()

    # Mention removal
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords removal
    tokens = [word for word in tokens if word not in stpwds_en]
    tokens = [word for word in tokens if word not in additional_stopwords]

    # Lemmatizing
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Combining Tokens
    text = ' '.join(tokens)

    return text

def run():
    # Membuat Judul
    st.title('Sentimen Analysis')

    # Membuat subheader
    st.subheader('Sentimen Analysis')

    user_input = st.text_area("Enter your text here:")

    if st.button('Predict'):
        # Preprocess the input text
        preprocessed_text = text_preprocessing(user_input)
        
        # Predict the sentiment
        predicted_sentiment = model.predict([preprocessed_text])
        predicted_sentiment = np.where(predicted_sentiment >= 0.5, 1, 0)

        # Menampilkan hasil prediksi
        if predicted_sentiment >= 0.5:
            st.write("This sentiment is a sentiment that likes McDonald's service.")
        else:
            st.write("This sentiment is a sentiment that does not like McDonald's service.")

if __name__ == '__main__':
    run()