## SentiFlix 
#### Movie Reviews Sentiment Analyzer using RNN

This project focuses on building a **binary classification model** to predict whether a movie review is **positive or negative** using deep learning techniques. The model leverages **word embeddings** and a **simple RNN** architecture.

### Project Overview

* **Dataset** : IMDb Dataset (Movie Reviews)
* **Records** : 50000+ movie reviews
* **Input Data** : Text reviews
* **Output Data** : Positive / Negative
* **Task Type** : Binary Classification

### Problem Statement

We aim to determine whether a movie review expresses a **positive** or **negative** sentiment.

The model is trained on the IMDb dataset, which contains thousands of movie reviews. Based on the input text our model predicts the sentiment class (positive or negative).

### Project Implementation Steps

```bash
IMDb Dataset
    ↓
Feature Engineering
    ↓
Training the Model(Simple RNN)
    ↓
Streamlit Web App
    ↓
Deployment
```

### Text Vectorization

To process text data we used a **Word Embedding** technique. This involves:

* Embedding Layer: Converts input words into dense vector representations.
* Word2Vec: Used as the word embedding technique.

```txt
Text → Word Embedding Layer → Dense Vectors → Simple RNN → Sigmoid → Output
```

### Model Architecture

* **Embedding Layer**: Transforms words into vectors
* **Activation**: Sigmoid (for binary output)
* **Loss Function**: Binary Crossentropy

### Technologies Used

* Python
* TensorFlow and Keras (for model building and training)
* Scikit-learn
* Streamlit (for web app)

### Deployment

The trained `.h5` model is integrated into a **Streamlit Web App** for real-time sentiment predictions. 

