import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

def preprocess_data(data):
    data['Jenis Kelamin'] = data['Jenis Kelamin'].replace({'PEREMPUAN': 0, 'LAKI-LAKI': 1})

    nama = [nama.lower().split() for nama in data['Nama']]
    word2vec_model = Word2Vec(nama, vector_size=100, window=7, min_count=1, workers=8)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(nama)

    return data, word2vec_model, tokenizer

def tokenize_sequences(data, tokenizer):
    X = data['Nama']
    y = data['Jenis Kelamin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(seq) for seq in X_train_sequences)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post')

    return X_train_padded, X_test_padded, y_train, y_test, max_len

def model_lstm(input_dim, output_dim, max_len):
   model = Sequential()
   model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_len))

   # Layer Bidirectional LSTM 1
   model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
   model.add(Dropout(0.5))

   # Layer Bidirectional LSTM 2
   model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
   model.add(Dropout(0.5))

   # Layer Bidirectional LSTM 3
   model.add(Bidirectional(LSTM(units=100)))
   model.add(Dropout(0.5))
   model.add(Dense(1, activation='sigmoid'))
   
   return model

def early_callback():
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-3,
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
    ]

    return callbacks

def main():
    data = pd.read_csv('dataset/data_gabungan.csv')
    data_preprocess, word2vec_mode, tokenizer = preprocess_data(data)
    X_train_padded, X_test_padded, y_train, y_test, max_len = tokenize_sequences(data, tokenizer)

    with open('model/tokenizer.pickle', 'wb') as handle:
        pickle.dump((tokenizer, max_len), handle, protocol=pickle.HIGHEST_PROTOCOL)

    input_dim = len(tokenizer.word_index) + 1
    output_dim = 100
    max_len = max(len(seq) for seq in X_train_padded)

    model = model_lstm(input_dim, output_dim, max_len)
    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )

    callback = early_callback()

    history = model.fit(
        X_train_padded,
        y_train,
        epochs=100,
        validation_data=[X_test_padded, y_test],
        callbacks=[callback]
    )

    model.save('model/model_modifikasi.h5')

    accuracy = model.evaluate(X_train_padded, y_train)[1]
    val_accuracy = model.evaluate(X_test_padded, y_test)[1]

    print('Training Accuracy: ', accuracy * 100, '%')
    print('Validation Accuracy: ', val_accuracy * 100, '%')


if __name__ == '__main__':
    main()