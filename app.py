## Import Libraries 
import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load word_index 
word_index = imdb.get_word_index()
reverse_word_index = { value:key for key,value in word_index.items()}

## Loading our trained RNN model 
model = load_model("simple_rnn.h5")

## Function to decode reviews 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## streamlit web app
st.title(":rainbow[Movie Analysis]")
st.subheader(":grey[Analyze movies based on text using simple RNN]")

## user input 
st.write(":rainbow[Enter the movie review to classify]")
user_review = st.text_area("Enter movie review")

## classification 
if st.button("classify"):
    ## preprocessing the user_review 
    preprocessed_user_input = preprocess_text(user_review)
    ## prediction 
    prediction = model.predict(preprocessed_user_input)
    ## sentiment 
    if prediction[0][0] > 0.5 :
        sentiment = "Positive"
    else :
        sentiment = "Negative"
    ## Display result 
    if sentiment is "Positive" :
        st.write(f"Sentiment : :green[{sentiment}⚡]")
    else :
        st.write(f"Sentiment : :red[{sentiment}😐]")
    st.write(f"Prediction score : :grey[{prediction[0][0]:2f}]")
else :
    st.error("please enter review")