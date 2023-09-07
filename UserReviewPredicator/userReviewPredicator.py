import numpy as np
import pandas as pd
import spacy
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Attention, Add
from sklearn.model_selection import train_test_split
from tqdm import tqdm

learning_rate = 0.001
dropout_rate = 0.2

class URP:
    def __init__(self, filename, reviewer_ID_idx, review_text_idx, review_rate_idx, max_user=None, epoch_no=None):
        self.filename = filename
        self.reviewer_ID_idx = reviewer_ID_idx
        self.review_text_idx = review_text_idx
        self.review_rate_idx = review_rate_idx
        self.max_user = max_user
        self.epoch_no = epoch_no

        self.rate_col_name = None
        self.text_col_name = None
        self.id_col_name = None
        self.data = None

        self.nlp = spacy.load('en_core_web_sm')
        self.tops_user = self.read_csv()
        self.target_users_data = self.process_users()

    def extract_emotion_words(self, text):
        doc = self.nlp(text)
        emotion_words = [token.text for token in doc if token.pos_ in ['ADJ', 'ADV']]
        return ' '.join(emotion_words)

    def read_csv(self):
        with open(self.filename) as file:
            line = file.readline()
        line = line.strip().split(',')
        self.id_col_name = line[self.reviewer_ID_idx]
        self.text_col_name = line[self.review_text_idx]
        self.rate_col_name = line[self.review_rate_idx]
        self.data = pd.read_csv(self.filename, usecols=[self.id_col_name, self.text_col_name, self.rate_col_name])

        return self.data[self.id_col_name].value_counts().nlargest(self.max_user).index

    def process_users(self):
        target_users_data = []
        for user in self.tops_user:
            # Filter the data to include only rows with the current user's reviewerID
            filtered_data = self.data[self.data[self.id_col_name] == user]

            # Assuming 'reviewText' contains the review text, and 'overall' contains the ratings
            review_text = filtered_data[self.text_col_name]
            ratings = filtered_data[self.rate_col_name]
            target_users_data.append({"review_text": review_text, "ratings": ratings})

        return target_users_data

    def get_predication_result(self, ratio):
        average_percentage_differences = []
        MAE_list = []
        MSE_list = []
        RMSE_list = []

        for user in tqdm(self.target_users_data):
            X_text_train, X_text_test, y_train, y_test = train_test_split(
                user["review_text"], user["ratings"], test_size=ratio, random_state=42
            )

            # emotion features extraction based on each review
            train_emotion, test_emotion = self.emotion_features_extraction(X_text_train, X_text_test)
            # text features extraction
            X_text_train_padded, X_text_test_padded = self.text_features_extraction(X_text_train, X_text_test)

            # model building
            model = self.modle_builder(train_emotion)

            # predication results
            y_pred = self.train_test(model, X_text_train_padded, X_text_test_padded, train_emotion, test_emotion, y_train)

            mae, mse, rmse, average_percentage_difference = self.evaluation(y_pred, y_test)
            average_percentage_differences.append(average_percentage_difference)
            MAE_list.append(mae)
            MSE_list.append(mse)
            RMSE_list.append(rmse)

        # average_percentage_difference
        apd = sum(average_percentage_differences)/self.max_user
        mae = sum(MAE_list)/self.max_user
        mse = sum(MSE_list)/self.max_user
        rmse = sum(RMSE_list)/self.max_user

        return apd, mae, mse, rmse

    def emotion_features_extraction(self, X_text_train, X_text_test):
        # extract sentiment entities from each review
        train_emotion_words = X_text_train.apply(self.extract_emotion_words)
        test_emotion_words = X_text_test.apply(self.extract_emotion_words)

        # emotion_features
        emotion_vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
        train_emotion_features = emotion_vectorizer.fit_transform(train_emotion_words)
        test_emotion_features = emotion_vectorizer.transform(test_emotion_words)

        train_emotion = np.array(train_emotion_features.todense())
        test_emotion = np.array(test_emotion_features.todense())

        return train_emotion, test_emotion

    def text_features_extraction(self, X_text_train, X_text_test):
        # create tokenizer
        tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_text_train)

        # convert text to seq
        X_text_train_seq = tokenizer.texts_to_sequences(X_text_train)
        X_text_test_seq = tokenizer.texts_to_sequences(X_text_test)

        # padding sequence
        X_text_train_padded = pad_sequences(X_text_train_seq, maxlen=100, padding='post', truncating='post')
        X_text_test_padded = pad_sequences(X_text_test_seq, maxlen=100, padding='post', truncating='post')

        return X_text_train_padded, X_text_test_padded

    def modle_builder(self, train_emotion):

        text_input = Input(shape=(100,), name='text_input')
        text_embedding = Embedding(input_dim=1000, output_dim=64)(text_input)
        emotion_input = Input(shape=(train_emotion.shape[1],), name='emotion_input')

        # add self-attention output to LSTM
        query_value_attention_seq = Attention()([text_embedding, text_embedding])
        text_combined = Add()([text_embedding, query_value_attention_seq])

        # LSTM
        text_lstm1 = LSTM(32, return_sequences=True)(text_combined)
        dropout1 = Dropout(dropout_rate)(text_lstm1)
        text_lstm2 = LSTM(64)(dropout1)
        dropout2 = Dropout(dropout_rate)(text_lstm2)

        concatenated = Concatenate()([dropout2, emotion_input])
        dense_layer = Dense(10, activation='relu')(concatenated)
        output = Dense(1, activation='linear')(dense_layer)
        model = Model(inputs=[text_input, emotion_input], outputs=[output])

        # Compile model with adaptive learning rate
        optimizer = Adam(learning_rate=learning_rate)
        # compile model
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        return model

    def train_test(self, model, X_text_train_padded, X_text_test_padded, train_emotion, test_emotion, y_train):
        # train
        model.fit(
            {'text_input': X_text_train_padded, 'emotion_input': train_emotion},
            y_train,
            epochs=self.epoch_no,
            validation_split=0.1,
            verbose=1
        )
        # predication
        y_pred = model.predict([X_text_test_padded, test_emotion])

        return y_pred

    def evaluation(self, y_pred, y_test):
        # calculate the percentage difference
        percentage_difference = np.abs((y_pred.reshape(-1) - y_test) / y_test) * 100
        # calculate the average percent difference
        average_percentage_difference = np.mean(percentage_difference)

        # calculate the mae, mse, and rmse
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)

        return mae, mse, rmse, average_percentage_difference