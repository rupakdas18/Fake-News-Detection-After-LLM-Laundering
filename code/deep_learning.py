# -*- coding: utf-8 -*-

#%%
import tensorflow as tf

# Print TensorFlow version and check GPU availability
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print("GPUs:", tf.config.list_physical_devices('GPU'))

#%%

import load_data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.linear_model import LogisticRegression

import nltk

from nltk.corpus import stopwords
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Dense, Dropout,Input

import pickle

# import evaluate


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('wordnet')








def plot_data(df):

  # Tokenize the text and count the number of words
  df['word_count'] = df['clean_text'].apply(lambda x: len(nltk.word_tokenize(x)))

  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  sns.histplot(df[df['label'] == 0]['word_count'], kde=True, color='skyblue')
  plt.title('Distribution of Word Count (Fake News)')
  plt.xlabel('Number of Words')
  plt.ylabel('Frequency')

  plt.subplot(1, 2, 2)
  sns.histplot(df[df['label'] == 1]['word_count'], kde=True, color='salmon')
  plt.title('Distribution of Word Count (True News)')
  plt.xlabel('Number of Words')
  plt.ylabel('Frequency')

  plt.tight_layout()
  plt.show()

    
def feature_label_split(df,text,label):

  X = df[text]
  y = df[label]

  list_of_words = []
  for i in X:
      for j in i.split():
          list_of_words.append(j)

  total_words = len(list(set(list_of_words)))
  print('Found %s unique tokens.' % total_words)
  print("Feature size = ", X.shape)
  print("label size = ", y.shape)

  return X, y, total_words



def cnn_classifier(X_train, X_test, X_val, y_train, y_test, y_val, num_class,vocabulary_size,max_text_len):

  # vocabulary_size = 15000
  # max_text_len = 768
  all_text = pd.concat([X_train, X_test])
  tokenizer = Tokenizer(num_words=vocabulary_size)
  tokenizer.fit_on_texts(all_text.values)
  
  # Save the tokenizer
  with open(f'{project}/Results/{classifier}/{dataset_name}_{paraphraser}_CNN_tokenizer.pickle', 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Convert texts to sequences
  train_sequences = tokenizer.texts_to_sequences(X_train.values)
  test_sequences = tokenizer.texts_to_sequences(X_test.values)
  val_sequences = tokenizer.texts_to_sequences(X_val.values)

    # Pad sequences to ensure uniform length
  # (This step is crucial for LSTM models)
  train_sequences = pad_sequences(train_sequences, maxlen=max_text_len)
  test_sequences = pad_sequences(test_sequences, maxlen=max_text_len)
  val_sequences = pad_sequences(val_sequences, maxlen=max_text_len)



    # One-hot encode the target variables
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_class)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_class)
  y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_class)



  # Setting up the parameters
  # maximum_features = vocabulary_size  # Maximum number of words to consider as features
  # #maximum_length = 100  # Maximum length of input sequences
  # word_embedding_dims = 50  # Dimension of word embeddings
  # no_of_filters = 250  # Number of filters in the convolutional layer
  # kernel_size = 3  # Size of the convolutional filters
  # hidden_dims = 250  # Number of neurons in the hidden layer
  # batch_size = 32  # Batch size for training
  # epochs = 10  # Number of training epochs
  # threshold = 0.5  # Threshold for binary classification

    # Tokenize the text data
  # tokenizer = Tokenizer(num_words=maximum_features)
  # tokenizer.fit_on_texts(X_train)

  # Convert text to sequences of integers
  # X_train = tokenizer.texts_to_sequences(X_train)
  # X_test = tokenizer.texts_to_sequences(X_test)
  # X_val = tokenizer.texts_to_sequences(X_val)


  # Padding the sequences to ensure uniform length
  # x_train = pad_sequences(X_train, maxlen=maximum_length)
  # x_test = pad_sequences(X_test, maxlen=maximum_length)
  # x_val = pad_sequences(X_val, maxlen=maximum_length)

  # Building the model
  # model = Sequential()

  # # Adding the embedding layer to convert input sequences to dense vectors
  # model.add(Embedding(maximum_features, word_embedding_dims,
  #                     input_length=max_text_len))

  # # Adding the 1D convolutional layer with ReLU activation
  # model.add(Conv1D(no_of_filters, kernel_size, padding='valid',
  #                 activation='relu', strides=1))

  # # Adding the global max pooling layer to reduce dimensionality
  # model.add(GlobalMaxPooling1D())

  # # Adding the dense hidden layer with ReLU activation
  # model.add(Dense(hidden_dims, activation='relu'))

  # # Adding the output layer with sigmoid activation for binary classification
  # model.add(Dense(num_class, activation='softmax'))

  # # Compiling the model with binary cross-entropy loss and Adam optimizer
  # model.compile(loss='categorical_crossentropy',
  #               optimizer='adam', metrics=['accuracy'])

  model = Sequential([
        Embedding(input_dim=vocabulary_size, output_dim=100, input_length=max_text_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=5),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_class, activation='sigmoid')  # Use 'softmax' and change units if more than two classes
    ])
    
  model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
   

  # Training the model
  model.fit(train_sequences, y_train, batch_size=32,
            epochs=10, validation_data=(val_sequences, y_val))


  # Predicting the probabilities for test data
  y_pred = model.predict(test_sequences)

  y_test_labels = np.argmax(y_test, axis=1)
  y_pred_labels = np.argmax(y_pred, axis=1)

  print(y_test_labels)
  print(type(y_pred))

  print(y_pred_labels)
  print(type(y_test))

  accuracy = accuracy_score(y_test_labels, y_pred_labels)
  precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
  recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
  f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')

  print(confusion_matrix(y_test_labels, y_pred_labels))
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1 Score: {f1}")

  test_df['CNN'] = y_pred_labels



    # Save the entire model
  model.save(f'{project}/Results/{classifier}/{dataset_name}_{paraphraser}_CNN.h5')

  # Save the model architecture to JSON
  model_json = model.to_json()
  with open(f'{project}/Results/{classifier}/{dataset_name}_{paraphraser}_CNN.json', 'w') as json_file:
      json_file.write(model_json)

  # Save the weights to HDF5
  model.save_weights(f'{project}/Results/{classifier}/{dataset_name}_{paraphraser}_CNN_model.weights.h5')

  print(model.summary())

  
  

    




from collections import Counter

def get_vocabulary_and_max_length(train_df, test_df, valid_df):
    # Concatenate all dataframes
    merged_df = pd.concat([train_df, test_df, valid_df])

    # Tokenize the text data
    all_text = merged_df['clean_text'].str.split()

    # Flatten the list of lists to get all tokens
    all_tokens = [word for sentence in all_text for word in sentence]

    # Get the vocabulary size
    vocabulary_size = len(set(all_tokens))

    # Get the maximum sentence length
    max_sentence_length = all_text.apply(len).max()

    return vocabulary_size, max_sentence_length





def lstm_classifier(X_train, X_test, X_val, y_train, y_test, y_val, num_class,vocabulary_size,max_text_len):

  all_text = pd.concat([X_train, X_test])
  tokenizer = Tokenizer(num_words=vocabulary_size)
  tokenizer.fit_on_texts(all_text.values)
  
  with open(f'Results/{dataset_name}_{paraphraser}_LSTM_tokenizer.pickle', 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Convert texts to sequences
  train_sequences = tokenizer.texts_to_sequences(X_train.values)
  test_sequences = tokenizer.texts_to_sequences(X_test.values)
  val_sequences = tokenizer.texts_to_sequences(X_val.values)

    # Pad sequences to ensure uniform length
  # (This step is crucial for LSTM models)
  train_sequences = pad_sequences(train_sequences, maxlen=max_text_len)
  test_sequences = pad_sequences(test_sequences, maxlen=max_text_len)
  val_sequences = pad_sequences(val_sequences, maxlen=max_text_len)



    # One-hot encode the target variables
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_class)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_class)
  y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_class)


  epochs = 10
  emb_dim = 256
  batch_size = 64
  model_lstm1 = Sequential()
  model_lstm1.add(Embedding(vocabulary_size,emb_dim, input_length=max_text_len))
  model_lstm1.add(SpatialDropout1D(0.8))
  model_lstm1.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5))
  model_lstm1.add(Dropout(0.5))
  model_lstm1.add(Flatten())
  model_lstm1.add(Dense(64, activation='relu'))
  model_lstm1.add(Dropout(0.5))
  model_lstm1.add(Dense(num_class, activation='softmax'))
  model_lstm1.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
  print(model_lstm1.summary())

  history_lstm1 = model_lstm1.fit(train_sequences, y_train, epochs = epochs, batch_size = batch_size, validation_data=(val_sequences,y_val))

  results_1 = model_lstm1.evaluate(test_sequences, y_test, verbose=False)
  print(f'Test results - Loss: {results_1[0]} - Accuracy: {100*results_1[1]}%')

  acc = history_lstm1.history['acc']
  val_acc = history_lstm1.history['val_acc']
  loss = history_lstm1.history['loss']
  val_loss = history_lstm1.history['val_loss']

  y_pred = model_lstm1.predict(test_sequences)

  y_test_labels = np.argmax(y_test, axis=1)
  y_pred_labels = np.argmax(y_pred, axis=1)

  print(y_test_labels)
  print(type(y_pred))

  print(y_pred_labels)
  print(type(y_test))

  accuracy = accuracy_score(y_test_labels, y_pred_labels)
  precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
  recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
  f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')

  print(confusion_matrix(y_test_labels, y_pred_labels))
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1 Score: {f1}")


  test_df['LSTM'] = y_pred_labels


  # Save the entire model
  model_lstm1.save(f'{project}/Results/{classifier}/{dataset_name}_{paraphraser}_LSTM.h5')

  # Save the model architecture to JSON
  model_json = model_lstm1.to_json()
  with open(f'{project}/Results/{classifier}/{dataset_name}_{paraphraser}_LSTM.json', 'w') as json_file:
      json_file.write(model_json)

  # Save the weights to HDF5
  model_lstm1.save_weights(f'{project}/Results/{classifier}/{dataset_name}_{paraphraser}_LSTM_model.weights.h5')


def lstm2(X_train,X_valid,y_train,y_valid):
    
    vocab_size = 14957
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, input_length=50, output_dim=4))
    model.add(Dropout(rate=0.4))
    model.add(LSTM(units=4))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=100,  activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=6, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    history = model.fit(
    X_train, 
    y_train, 
    validation_data=[X_valid, y_valid],
    epochs = 5
    )






if __name__=="__main__":



    dataset_name = 'covid-19' # 'TALLIP''liar_2','liar_6', 'kaggle', 'covid-19'
    language = "english"
    num_class = 2 # 2, 6
    paraphraser = 'human' # 'human', 'bard', 'parrot', 'gpt', 'pegasus','llama'
    classifier = 'CNN' # 'lstm', 'cnn'
    project = "02-Paraphrasing"
    

    
    full_df, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_class,paraphraser,language,classifier)
    
    
    
    print("Shape before removing NAN values")
    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)
    
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    valid_df = valid_df.dropna()
    
    print("Shape after removing NAN values")
    
    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)

    
    
    text_length = 1200
    
    train_df['clean_text']=train_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
    test_df['clean_text']=test_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
    valid_df['clean_text']=valid_df['text'].apply(lambda x: load_data.clean_text(x,text_length))

    
    
    vocabulary_size, max_text_len = get_vocabulary_and_max_length(train_df, test_df, valid_df)
    
    print("Vocabulary size: ", vocabulary_size)
    print("Max text length: ", max_text_len)
    
    
    X_train = train_df['clean_text']
    Y_train = train_df['label']
    X_test = test_df['clean_text']
    Y_test = test_df['label']
    X_val = valid_df['clean_text']
    Y_val = valid_df['label']
    
    
    print("Labels: ", Y_train.unique())
    print(X_train.values)
    
      
    
    #ros = RandomOverSampler(random_state=42)
    #X_train, Y_train = ros.fit_resample(X_train, Y_train)
    
    if classifier == 'CNN':
      cnn_classifier(X_train, X_test, X_val, Y_train, Y_test, Y_val,num_class,vocabulary_size,max_text_len)
      test_df.to_excel(f"{project}/Results/{classifier}/{dataset_name}_{paraphraser}_CNN_results.xlsx")

    
    elif classifier == 'LSTM':
      lstm_classifier(X_train, X_test, X_val, Y_train, Y_test, Y_val,num_class,vocabulary_size,max_text_len)
      test_df.to_excel(f"{project}/Results/{classifier}/{dataset_name}_{paraphraser}_LSTM_results.xlsx")



# %%
