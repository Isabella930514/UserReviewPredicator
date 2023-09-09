# User Review Prediction (URP)
This project aims to predict the rating of reviews based on the text and extracted emotional features. I am focusing on the top reviewers, given that they have the most reviews (also can be used for all users in the dataset). I use self-attention-based LSTM (Long Short-Term Memory) networks, TF-IDF, and a fully connected neural network for this prediction.

# Data Set:
This data set is a virtual reviewer interaction record, which is generated by ChatGPT. You may need to import and convert your data set inside.

# Tensorflow, LSTM, Self Attention, NLP (Sentiment entities extraction)
# Outline:

Data Loading
Data Pre-processing
Model Building
Model Training and Validation
Model Evaluation
Results and Visualizations

# Assumptions:
  In order to design a satisfactory model within a limited timeframe, I made some assumptions while creating the model. These assumptions are as follows:
  1. One account is for use by one person only.
  2. A reviewer's behavior in commenting does not change over time (i.e., when someone comments 'good,' they genuinely mean it's 'good,' and this won't change over time to mean 'it's just okay').

# Description:

1. Collecting reviewers' ID, historical review content, and rate.
2. Using The Keras Tokenizer to convert each word of textual reviews to its position in the vocabulary, and padding to a specific size.
3. The extract_emotion_words function aims to extract words that are likely to convey emotions from a given text. The function assumes that adjectives (ADJ) and adverbs (ADV) are good indicators of emotions, which is often true in natural language.
4. Using the TF-IDF Vectorizer from the scikit-learn library to convert a list of "emotion words" into numerical vectors.
5. Using the self-attention model to describe the importance of each word in sentences.
6. The textual data is processed using an LSTM model enhanced with attention mechanisms, while the emotional aspects are captured using a TF-IDF model for embedding.
7. These two types of features are then fused and trained using a fully connected neural network to make the final prediction.

# Run:
python main.py

