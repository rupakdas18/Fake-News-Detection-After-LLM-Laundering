# -*- coding: utf-8 -*-
# Performance Analysis between Machine Learning Classifiers for Fake News Detection (Supervised Learning and BERT)

## Import Libraries
"""

# Import libraries
# !pip install gensim -q
# !pip install wordcloud -q
# !pip install mlxtend -q
# !pip install imblearn -q
# !pip install keras -q
# !pip install playsound


import load_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import get_blas_funcs
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from keras import layers
from sklearn.svm import SVC
from sklearn import svm
import xgboost as xgb
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from keras.models import Sequential
#from keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
import gensim.downloader as api
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from transformers import AutoTokenizer
#from tensorflow.keras.layers import Dense, Dropout,Input
#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.optimizers import Adam
from mlxtend.plotting import plot_confusion_matrix
import random
import torch
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import gensim
import gensim.downloader as api
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import os
import functools
import torch
import torch.nn.functional as F
# import evaluate

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

# from datasets import Dataset, DatasetDict
# from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from playsound import playsound
nltk.download('punkt_tab')



"""### Create a alarm"""

from IPython.display import Audio
def generate_alarm():
    print("Generating alarm sound...")
    audio_file = '/content/drive/MyDrive/597_dataset/alarm.mp3'  # Replace with your file path
    return Audio(audio_file, autoplay=True)

"""## Load dataset

### Load Kagle dataset
"""

# Connect with google drive and load data



try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('wordnet')

def clean_text(text,word_count):
    """Process text function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    text= re.sub('\[[^]]*\]', '', text)
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    #removal of html tags
    review =re.sub(r'<.*?>',' ',text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = re.sub("["
                           u"\U0001F600-\U0001F64F"  # removal of emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+",' ',text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text_tokens =word_tokenize(text)

    text_clean = []
    for word in  text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            lem_word =lemmatizer.lemmatize(word)  # lemmitiging word
            text_clean.append(lem_word)
    text_mod=[i for i in text_clean if len(i)>2]
    text_clean=' '.join(text_mod)

    words = text_clean.split()
    first_n_words = words[:word_count]
    trunced_clean_text = ' '.join(first_n_words)


    return  trunced_clean_text

"""## Feature Engineering"""

# TF-IDF

def tf_idf (X_train, X_test, X_val, total_words):

  vectorization = TfidfVectorizer(max_features=total_words)
  tf_X_train = vectorization.fit_transform(X_train)
  tf_X_test = vectorization.transform(X_test)
  tf_X_val = vectorization.transform(X_val)

  return tf_X_train, tf_X_test, tf_X_val, vectorization

# CountVectorizer

def counter_vector(X_train, X_test, X_val, total_words):

  countv = CountVectorizer(max_features=total_words)
  cv_X_train = countv.fit_transform(X_train)
  cv_X_test = countv.transform(X_test)
  cv_X_val = countv.transform(X_val)

  return cv_X_train, cv_X_test, cv_X_val, countv

# Word Embeddings

def word_embeddings(X,y,word2vec_model):

# Run this if you do not have a saved file.
  # try:
  #   word2vec_model = api.load('word2vec-google-news-300')
  #   print("Google News Word2Vec model downloaded successfully.")
  # except Exception as e:
  #   print(f"Error downloading the model:

  #corpus = X.to_numpy()
  #labels = y.to_numpy()
  corpus = X
  labels = y

  # Tokenize the sentences into words
  tokenized_corpus = [sentence.split() for sentence in corpus]

  # Load pre-trained Word2Vec model (you can download a pre-trained model or train your own)
  # Example using pre-trained Word2Vec model from gensim's KeyedVectors

  # Transform each sentence into an average word vector
  def sentence_to_avg_vector(sentence, model):
      vectors = [model[word] for word in sentence if word in model]
      return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

  X_word2vec = np.array([sentence_to_avg_vector(sentence, word2vec_model) for sentence in tokenized_corpus])

  # Split data for training and testing

  return X_word2vec, labels



import numpy as np
def glove_embeddings():


  # Path to the GloVe file
  glove_file_path = "glove.42B.300d.txt"

  # Initialize an empty dictionary to store word embeddings
  embeddings_index = {}

  # Load the GloVe embeddings
  with open(glove_file_path, 'r', encoding='utf-8') as f:
      for line in f:
          # Split each line into the word and its vector
          values = line.split()
          word = values[0]  # The word
          vector = np.asarray(values[1:], dtype='float32')  # The vector as a numpy array
          embeddings_index[word] = vector

  print(f"Loaded {len(embeddings_index)} word vectors.")

  return embeddings_index


def preprocess_text(X, embeddings_index):
    """
    Transform each document into a feature vector by averaging the GloVe vectors for all words in the document.
    """
    embedding_dim = len(next(iter(embeddings_index.values())))  # Get the dimension of the embeddings
    X_vectors = []

    for doc in X:
        words = doc.split()  # Tokenize the document into words
        valid_vectors = [embeddings_index[word] for word in words if word in embeddings_index]

        if valid_vectors:
            # Compute the mean of the word vectors
            doc_vector = np.mean(valid_vectors, axis=0)
        else:
            # If no words in the document have embeddings, use a zero vector
            doc_vector = np.zeros(embedding_dim)

        X_vectors.append(doc_vector)

    return np.array(X_vectors)

"""###BERT word-embeddings"""

# Function to encode text and extract embeddings


def get_bert_embeddings(text,tokenizer, model):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    # Get embeddings from the last hidden state (mean pooling)
    embeddings = output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

"""### ELMo word-embeddings"""

import tensorflow_hub as hub

# def get_elmo_embeddings(sentences):
#     elmo = hub.load("https://tfhub.dev/google/elmo/2")
#     embeddings = elmo.signatures['default'](tf.constant(sentences))['elmo']
#     return embeddings.numpy()



def get_elmo_embeddings(sentences, elmo, batch_size=32):
    
    total_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)
    all_embeddings = []

    for i in range(total_batches):
        batch_sentences = sentences[i * batch_size:(i + 1) * batch_size]
        batch_embeddings = elmo.signatures['default'](tf.constant(batch_sentences))['elmo']
        all_embeddings.append(batch_embeddings)

    # Pad the embeddings to have the same sequence length
    max_sequence_length = max(emb.shape[1] for emb in all_embeddings)
    padded_embeddings = [tf.pad(emb, [[0, 0], [0, max_sequence_length - emb.shape[1]], [0, 0]]) for emb in all_embeddings]

    # Concatenate all batch embeddings
    concatenated_embeddings = tf.concat(padded_embeddings, axis=0)
    return concatenated_embeddings.numpy()

"""### Plot Data"""

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

"""### Option for Oversampling"""

def oversampling(X,y):
  # Oversampling
  smote = SMOTE(random_state=42)
  X_resampled, Y_resampled = smote.fit_resample(X, y)

  return X_resampled, Y_resampled

"""## Supervised classifiers"""

# Classification

def naive_bayes():
  nb = MultinomialNB()
  return nb

# Logistic Regression

def logistic_regression():
  lr = LogisticRegression(max_iter=1000)
  return lr

def decision_tree():
  dt = DecisionTreeClassifier()
  return dt

def svm_classifier():
  #svm = SGDClassifier()
  clf2 = svm.SVC(kernel='linear', C=1,probability=True)
  return clf2

def random_forest():
  rfc = RandomForestClassifier(random_state=0)
  return rfc

def xgboost_classifier():
  xgboost = xgb.XGBClassifier()
  return xgboost

"""### Find the performance"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import joblib

#import shap


#scoring_methods = ['accuracy', 'precision', 'recall', 'f1']
scoring_methods = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro'
}

def cross_val(model,train_X,test_X, Y_train,Y_test,model_name):
  #kf = KFold(n_splits=5, shuffle=True, random_state=42)
  #cv_results = cross_validate(model, tf_X, y, cv=kf, scoring=scoring_methods)

  model.fit(train_X, Y_train)
  y_pred = model.predict(test_X)


  print(classification_report(Y_test, y_pred))
  print(confusion_matrix(Y_test, y_pred))

  accuracy = accuracy_score(Y_test, y_pred)
  precision = precision_score(Y_test, y_pred, average='weighted')
  recall = recall_score(Y_test, y_pred, average='weighted')
  f1 = f1_score(Y_test, y_pred, average='weighted')


  classification_results[model_name] = y_pred

  # Save the model to a file
  filename = f'results/{dataset_choice}_{paraphraser}_{feature_choice}_{model_name}.pkl'
  joblib.dump(model, filename)

  return accuracy,precision,recall, f1

"""### Get the TF-IDF results"""

def get_classification_results(X,y,train_X,test_X, val_X, Y_train,Y_test, Y_val,feature_choice):

  print(f"Working with {feature_choice} feature..............................................................")

  tf_idf_results = pd.DataFrame(columns=['Model','Accuracy', 'F1', 'Precision','Recall'])

#   accuracy,precision,recall,f1 = cross_val(naive_bayes(),train_X,test_X, Y_train,Y_test,'NB')
#   row1 = {'Model': 'Naive Bayes', 'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
#   tf_idf_results = tf_idf_results._append(row1, ignore_index=True)
#   print("Done with Naive Bayes!!!!!!!!")

  accuracy,precision,recall,f1 = cross_val(logistic_regression(),train_X,test_X, Y_train,Y_test,'LR')
  row2 = {'Model': 'Logistic regression', 'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
  tf_idf_results = tf_idf_results._append(row2, ignore_index=True)
  print("Done with Logistic Regression!!!!!!!!")


  accuracy,precision,recall,f1 = cross_val(decision_tree(),train_X,test_X, Y_train,Y_test,'DT')
  row3 = {'Model': 'Decision tree', 'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
  tf_idf_results = tf_idf_results._append(row3, ignore_index=True)
  print("Done with Decision Tree!!!!!!!!")

  accuracy,precision,recall,f1 = cross_val(random_forest(),train_X,test_X, Y_train,Y_test,'RF')
  row4 = {'Model': 'Random Forest', 'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
  tf_idf_results = tf_idf_results._append(row4, ignore_index=True)
  print("Done with Random Forest!!!!!!!!")

  accuracy,precision,recall,f1 = cross_val(svm_classifier(),train_X,test_X, Y_train,Y_test,'SVM')
  row5 = {'Model': 'SVM', 'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
  tf_idf_results = tf_idf_results._append(row5, ignore_index=True)
  print("Done with SVM !!!!!!!!")

  accuracy,precision,recall,f1 = cross_val(xgboost_classifier(),train_X,test_X, Y_train,Y_test,'xgboost')
  row5 = {'Model': 'xgboost', 'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
  tf_idf_results = tf_idf_results._append(row5, ignore_index=True)
  print("Done with Xgboost !!!!!!!!")


  print(tf_idf_results)
  #tf_idf_results.to_csv('results/tfidf_results.csv', index=False)
  tf_idf_results.to_csv(f'results/{dataset_choice}_{paraphraser}_{feature_choice}.csv', index=False)
  return tf_idf_results



def count_unique_words(df):
    # Combine all the text in the specified column into a single string
    all_text = ' '.join(df["clean_text"].tolist())

    # Split the text into words
    words = all_text.split()

    # Get the set of unique words
    unique_words = set(words)

    # Return the number of unique words
    return len(unique_words)



def combine_features(X_train,Y_train,X_test,Y_test,X_val,Y_val, features):
    # Initialize an empty list to hold the feature arrays
    train_feature_arrays = []
    test_feature_arrays = []
    val_feature_arrays = []
    
    if 'tfidf' in features:

        tfidf_vectorizer = TfidfVectorizer(max_features=total_words)
        tf_X_train = tfidf_vectorizer.fit_transform(X_train)
        tf_X_test = tfidf_vectorizer.transform(X_test)
        tf_X_val = tfidf_vectorizer.transform(X_val)

        train_feature_arrays.append(tf_X_train.toarray())
        test_feature_arrays.append(tf_X_test.toarray())
        val_feature_arrays.append(tf_X_val.toarray())

    
    if 'cv' in features:
        countv = CountVectorizer(max_features=total_words)
        cv_X_train = countv.fit_transform(X_train)
        cv_X_test = countv.transform(X_test)
        cv_X_val = countv.transform(X_val)

        train_feature_arrays.append(cv_X_train.toarray())
        test_feature_arrays.append(cv_X_test.toarray())
        val_feature_arrays.append(cv_X_val.toarray())
    
    if 'w2v_we' in features:
        # Assuming function to convert data to averaged word vectors is defined elsewhere
        #word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("/content/drive/MyDrive/597_dataset/word2vec-google-news-300.bin", binary=True)
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec-google-news-300.bin", binary=True)
        print("Sucessfully loaded the Word2vec model")

        X_train_we, Y_train_we = word_embeddings(X_train,Y_train,word2vec_model)
        X_test_we, Y_test_we = word_embeddings(X_test,Y_test,word2vec_model)
        X_val_we, Y_val_we = word_embeddings(X_val,Y_val,word2vec_model)

        train_feature_arrays.append(X_train_we)
        test_feature_arrays.append(X_test_we)
        val_feature_arrays.append(X_val_we)


    
    if 'glove_we' in features:

        embeddings_index = glove_embeddings()
        print(type(embeddings_index))
        X_train_we = preprocess_text(X_train, embeddings_index)
        X_test_we = preprocess_text(X_test, embeddings_index)
        X_val_we = preprocess_text(X_val, embeddings_index)

        Y_train_we = Y_train
        Y_test_we = Y_test
        Y_val_we = Y_val
        # Load GloVe model as shown in your original code

        train_feature_arrays.append(X_train_we)
        test_feature_arrays.append(X_test_we)
        val_feature_arrays.append(X_val_we)

    
    if 'bert_we' in features:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        X_train_embds = [get_bert_embeddings(text,tokenizer, model) for text in X_train]
        X_test_embds = [get_bert_embeddings(text,tokenizer, model) for text in X_test]
        X_val_embds = [get_bert_embeddings(text,tokenizer, model) for text in X_val]

        X_train_we = np.vstack(X_train_embds)
        X_test_we = np.vstack(X_test_embds)
        X_val_we = np.vstack(X_val_embds)

        Y_train_we = Y_train
        Y_test_we = Y_test
        Y_val_we = Y_val

        train_feature_arrays.append(X_train_we)
        test_feature_arrays.append(X_test_we)
        val_feature_arrays.append(X_val_we)
    
    if 'elmo_we' in features:
        elmo = hub.load("https://tfhub.dev/google/elmo/3")
        print("Successfully loaded the ELMo model")

        X_train_embds = get_elmo_embeddings(X_train,elmo)
        X_test_embds = get_elmo_embeddings(X_test,elmo)
        X_val_embds = get_elmo_embeddings(X_val,elmo)


        X_train_we = np.mean(X_train_embds, axis=1)
        X_test_we = np.mean(X_test_embds, axis=1)
        X_val_we = np.mean(X_val_embds, axis=1)

        Y_train_we = Y_train
        Y_test_we = Y_test
        Y_val_we = Y_val

        train_feature_arrays.append(X_train_we)
        test_feature_arrays.append(X_test_we)
        val_feature_arrays.append(X_val_we)

    # Concatenate all feature arrays along the features axis
    combined_features_train = np.hstack(train_feature_arrays)
    combined_features_test = np.hstack(test_feature_arrays)
    combined_features_val = np.hstack(val_feature_arrays)

    return combined_features_train, combined_features_test, combined_features_val


"""## Main Code"""

import time
# Data load. Use this if you use the raw data
#df = load_data()

# Data cleaning. Use this if you need to do the data processing
#df['clean_text']=df['text'].apply(lambda x: clean_text(x))

# Use cleaned data. Use this line if you do not want to load data or do data pre-processing


dataset_choice = 'covid-19' # kaggle, liar_2, liar_6, covid-19, TALLIP
language = 'English'

num_classes = 2
feature_choice = ['elmo_we']  # User specifies the features they want to combine # tfidf, cv, w2v_we, bert_we, elmo_we, glove_we
raw_clean_choice = 'raw' # raw, clean
balance_choice = 'no' # no,yes
text_length = 5000
paraphraser = 'human' #  'parrot' or 'bard', 'gpt','pegasus','llama'




full_df, train_df, test_df, valid_df =  load_data.data_load(dataset_choice,num_classes,paraphraser,language)



# Use a smaller dataset to check the code.
# train_df = train_df.head(250)
# test_df = test_df.head(120)
# valid_df = valid_df.head(120)



print("Shape of training dataset before removing NA", train_df.shape)
print("Shape of testing dataset before removing NA", test_df.shape)
print("Shape of validation dataset before removing NA", valid_df.shape)

train_df = train_df.dropna()
test_df = test_df.dropna()
valid_df = valid_df.dropna()


print("Shape of training dataset after removing NA", train_df.shape)
print("Shape of testing dataset after removing NA", test_df.shape)
print("Shape of validation dataset after removing NA", valid_df.shape)



train_df['clean_text']=train_df['text'].apply(lambda x: clean_text(x,text_length))
test_df['clean_text']=test_df['text'].apply(lambda x: clean_text(x,text_length))
valid_df['clean_text']=valid_df['text'].apply(lambda x: clean_text(x,text_length))



total_words = count_unique_words(train_df)

X = full_df['text'].astype(str)
y = full_df['label']

#X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state =42)


X_train = train_df['clean_text']
Y_train = train_df['label']
X_test = test_df['clean_text']
Y_test = test_df['label']
X_val = valid_df['clean_text']
Y_val = valid_df['label']


print("Shape of training feature: ", X_train.shape)
print("Shape of training labels: ", Y_train.shape)
print("Shape of testing feature: ", X_test.shape)
print("Shape of testing labels: ", Y_test.shape)
print("Shape of validation feature: ", X_val.shape)
print("Shape of validation labels: ", Y_val.shape)



classification_results = pd.DataFrame({
    'Test_Features': X_test,
    'True_Labels': Y_test
})



X_train_com, X_test_com, X_val_com  = combine_features(X_train,Y_train,X_test,Y_test,X_val,Y_val,feature_choice)
print("Combined features shape: ", X_train_com.shape, X_test_com.shape, X_val_com.shape)
print(type(X_train_com))
df=pd.DataFrame(data=X_train_com[0:,0:],
          index=[i for i in range(X_train_com.shape[0])],
          columns=['f'+str(i) for i in range(X_train_com.shape[1])])
print(df.head)
df.to_csv('results/combined_features.csv', index=False)

result = get_classification_results(X, y, X_train_com, X_test_com, X_val_com, Y_train, Y_test, Y_val,feature_choice)



classification_results.to_excel(f"results/{dataset_choice}_{paraphraser}_{feature_choice}_classification_results.xlsx")
#generate_alarm()



