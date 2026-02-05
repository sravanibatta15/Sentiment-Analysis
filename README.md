# Sentiment-Analysis
Project Overview:

This project implements an end-to-end Sentiment Analysis System using Natural Language Processing (NLP) and Deep Learning. The system classifies movie reviews as positive or negative based on their textual content. It uses a Bi-Directional Recurrent Neural Network (RNN) built with TensorFlow and Keras to learn contextual meaning from text in both forward and backward directions.

The project demonstrates a complete pipeline starting from data loading and text preprocessing to model training, evaluation, and real-time prediction on new reviews.

Objectives:

The main goals of this project are:

• To clean and preprocess raw text data

• To convert text into numerical representations using embeddings

• To build a Bi-Directional RNN for sentiment classification

• To train and validate the model on IMDB movie reviews

• To perform real-time sentiment prediction on new input text

 Technologies Used:

• Programming Language: Python

• Libraries: NumPy, Pandas, NLTK, Scikit-learn

• Deep Learning Framework: TensorFlow / Keras

• NLP Tools: NLTK (stopwords, lemmatization)

• Model Type: Bi-Directional RNN

Dataset:

The project uses the IMDB Movie Reviews Dataset, which contains labeled movie reviews with sentiments:

• Positive

• Negative

Only the first 10,000 samples are used in this implementation for training and evaluation.

Processing Pipeline:

The system follows these steps:

Load the dataset from CSV

Map sentiment labels (negative → 0, positive → 1)

Clean text data

Convert to lowercase

Remove punctuation

Remove stopwords

Apply lemmatization

Convert text into numerical sequences (word embeddings)

Pad sequences to a fixed length

Split data into training, validation, and test sets

Train a Bi-Directional RNN model

Save the trained model

Predict sentiment for new input reviews

Key components:

• Embedding layer

• Bi-Directional RNN layer

• Dense output layer with sigmoid activation

How to Run the Project:

Install required libraries

pip install numpy pandas nltk scikit-learn tensorflow


Download NLTK resources:

import nltk

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('omw-1.4')


Run the project:

python main.py

Sample Prediction

Input Review:

The film starts with an interesting premise but quickly loses its way. The storyline feels predictable and stretched, with very little emotional impact. Performances are flat, and the pacing makes even important scenes feel dull. Overall, it’s a movie that had potential but fails to deliver.

Output:

Negative

Applications:

This project can be used for:

• Movie review analysis

• Product review sentiment detection

• Customer feedback analysis

• Opinion mining

• Social media sentiment tracking

Conclusion:

This project demonstrates how NLP and Deep Learning can be combined to build an intelligent sentiment analysis system. By using a Bi-Directional RNN, the model captures contextual meaning from text more effectively than traditional methods. The modular design allows easy extension to other datasets and real-world applications.

Author:

B. Siva Sai Sravani

Data Science / AI-ML

Email: sivasaisravani@gmail.com

LinkedIn: https://www.linkedin.com/in/siva-sai-sravani/

GitHub: https://github.com/sravanibatta15
