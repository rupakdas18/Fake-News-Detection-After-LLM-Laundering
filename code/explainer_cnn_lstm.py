# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import joblib
import os
import functools
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

# from datasets import Dataset, DatasetDict
# from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
import gensim.downloader as api
import gensim
import gensim.downloader as api
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import glob
import pickle

#%%

'''
Deep Learning

'''

"""## Load already classified dataset"""

# Choose dataset
dataset_name = 'liar_6'  #'kaggle','liar_6','covid-19'
paraphraser = 'no' 
idx = 1240
deep_learning = 'LSTM'


directory_cnn = '02-Paraphrasing/Results/' + dataset_name + '/CNN/'
directory_lstm = '02-Paraphrasing/Results/' + dataset_name + '/LSTM/'

directory_cnn = f'02-Paraphrasing/Results/{dataset_name}_{paraphraser}_CNN'
directory_lstm = f'02-Paraphrasing/Results/{dataset_name}_{paraphraser}_LSTM'


print(directory_cnn)
print(directory_lstm)


dataframe_cnn = pd.read_excel(f'02-Paraphrasing/Results/{dataset_name}_{paraphraser}_CNN_results.xlsx')
dataframe_lstm = pd.read_excel(f'02-Paraphrasing/Results/{dataset_name}_{paraphraser}_LSTM_results.xlsx')


print(dataframe_lstm['text'][86])

from tensorflow.keras.models import load_model


models_cnn = load_model(f'{directory_cnn}.h5')
models_lstm = load_model(f'{directory_lstm}.h5')

print(models_cnn)
print(models_lstm)


print(models_cnn.summary())
print(models_lstm.summary())

if paraphraser == 'no':
    max_length = 321
    
else:
    layer = models_cnn.get_layer(name='embedding')
    output_shape = layer.output_shape
    max_length = output_shape[1]


"""### Load the pre-trained model"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# all_text = pd.concat([X_train, X_test,X_val])
# tokenizer = Tokenizer(num_words=vocabulary_size)
# #tokenizer.fit_on_texts(X_train.values)
# tokenizer.fit_on_texts(all_text.values)

# # Save the tokenizer
# with open('liar_dL_tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if dataset_name == 'kaggle':

  max_text_len = 4500
  with open(f'02-Paraphrasing/Results/{dataset_name}_{paraphraser}_LSTM_tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
      
elif dataset_name == 'liar_6':
  max_text_len = max_length
  with open(f'02-Paraphrasing/Results/{dataset_name}_{paraphraser}_LSTM_tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
      
elif dataset_name == 'covid-19':
    max_text_len = max_length
    with open(f'02-Paraphrasing/Results/{dataset_name}_{paraphraser}_LSTM_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)



def predict_fn_cnn(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_text_len)
    return models_cnn.predict(padded)


def predict_fn_lstm(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_text_len)
    return models_lstm.predict(padded)

from lime.lime_text import LimeTextExplainer


#test_text = "Historically, the U.S. Supreme Court has not been known to pose many questions during oral arguments."
test_text = "The U.S. Supreme Court has not traditionally asked a lot of questions during oral arguments."
#test_text = dataframe_lstm.clean_text[idx]

if deep_learning == 'CNN':

    if dataset_name == 'covid-19':
      class_names = ["Fake", "True"]
      print("True class:",  class_names[int(dataframe_cnn.label[idx])])
      explainer = LimeTextExplainer(class_names=class_names)
      explanation = explainer.explain_instance(dataframe_cnn.clean_text[idx], predict_fn_cnn, num_features=10)
    
    
    elif dataset_name == 'liar_6' or dataset_name == 'liar_2':
      class_names = ["pants-fire", "false","barely-true","half-true","mostly-true","true"]
      explainer = LimeTextExplainer(class_names=class_names)
      explanation = explainer.explain_instance(dataframe_cnn.clean_text[idx], predict_fn_cnn, num_features=10, labels=[0, 1,2,3,4,5])
      print("True class:",  class_names[int(dataframe_cnn.label[idx])])
    explanation.show_in_notebook(text=True)
    explanation.save_to_file(f'02-Paraphrasing/Figures/{dataset_name}_{paraphraser}_CNN_sample_{idx}.html')
    


if deep_learning == 'LSTM':
    # Display the explanation
    # print(explanation.as_list())
   
    
    if dataset_name == 'covid-19':
      class_names = ["Fake", "True"]
      print("True class:",  class_names[int(dataframe_lstm.label[idx])])
      explainer = LimeTextExplainer(class_names=class_names)
      explanation = explainer.explain_instance(test_text, predict_fn_lstm, num_features=10,top_labels=1)
    
    
    elif dataset_name == 'liar_6' or dataset_name == 'liar_2':
      class_names = ["pants-fire", "false","barely-true","half-true","mostly-true","true"]
      explainer = LimeTextExplainer(class_names=class_names)
      explanation = explainer.explain_instance(test_text, predict_fn_lstm, num_features=10,top_labels=1)
      print(explanation.available_labels())
      print("True class:",  class_names[int(dataframe_lstm.label[idx])])
    

    #print(explanation.as_list())
    explanation.show_in_notebook(text=True)
    explanation.save_to_file(f'02-Paraphrasing/Figures/{dataset_name}_{paraphraser}_LSTM_sample_{idx}.html')


#%%

'''
Supervised Learning

'''

"""## Load already classified dataset"""

import numpy as np

# Choose dataset
dataset_name = 'liar_6'  #'kaggle','liar_6','covid-19'
paraphraser = 'llama' 
feature_name = 'wv' # 'tfidf', 'cv', 'wv'
idx = 882


directory = f'02-Paraphrasing/Results/supervised/'
print(directory)

test_dataframe = pd.read_excel(f"02-Paraphrasing/Results/supervised/{dataset_name}_{paraphraser}_{feature_name}_classification_results.xlsx")


pattern = f"{dataset_name}_{paraphraser}_{feature_name}_*.pkl"
filename = directory + pattern

print(filename)

# Get a list of all files that match the pattern
model_files = glob.glob(filename)
models = [joblib.load(model_file) for model_file in model_files]

print(f"Loaded {feature_name} feature for {dataset_name} dataset")
print(models)



if feature_name == 'tfidf' or feature_name == 'cv':

    with open(f'02-Paraphrasing/Results/supervised/{dataset_name}_{paraphraser}_{feature_name}_vectorizer.pickle', 'rb') as handle:
      vectorization = pickle.load(handle)
    print(f"{feature_name} tokenizer for {dataset_name}_{paraphraser} dataset is loaded")



elif  feature_name == 'wv':
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(f"02-Paraphrasing/Results/supervised/word2vec-google-news-300.bin", binary=True)
    print("Sucessfully loaded the Word2vec model")



from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin


# Function to get word embeddings
def get_word2vec_embedding(text):
    # print("Inside the class function")
    # print(text)
    words = text.split()
    word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(300)


# Using LIME for explanation
class Word2VecEmbedding:
    def __init__(self, model):
        self.model = model

    def transform(self, texts):
        return np.array([get_word2vec_embedding(text) for text in texts])




# for n in range (5,6):

#   idx = filtered_df.index[n]
#   print(n, idx)


#models = [models[2]]

for model in models:

  print("Loaded model: ", model)
  #print(type(model))



  #print(filtered_df.iloc[[idx]])
  #print(filtered_df.iloc[idx]["Test_Features"])
  #print(filtered_df["Test_Features"][idx])
  #print(filtered_df.Test_Features[idx])
  if feature_name == 'wv':
    word2vec_transformer = Word2VecEmbedding(word2vec_model)
    c = make_pipeline(word2vec_transformer, model)
  else:
    c = make_pipeline(vectorization, model)

  if dataset_name == 'covid-19':

    class_names = ["Fake", "True"]

    probabilit_list = (c.predict_proba([test_dataframe["Test_Features"][idx]]).round(3))
    probabilit_list = list(probabilit_list[0])
    explainer = LimeTextExplainer(class_names = class_names)
    exp = explainer.explain_instance(test_dataframe["Test_Features"][idx], c.predict_proba, num_features = 10)
    print ("True class:",  class_names[int(test_dataframe.True_Labels[idx])])
    #print(c.predict_proba([test_dataframe.text[idx]]).round(4))
    #print("Probability (Fake) =", c.predict_proba([test_dataframe["text"][idx]])[0, 0])
    #print("Probability (True) =", c.predict_proba([test_dataframe["text"][idx]])[0, 1])


  elif dataset_name == 'liar_6':

    class_names = ["pants-fire", "false","barely-true","half-true","mostly-true","true"]
    probabilit_list = (c.predict_proba([test_dataframe.Test_Features[idx]]).round(3))
    probabilit_list = list(probabilit_list[0])
    max_prob_index = int(probabilit_list.index(max(probabilit_list)))
    #print(max_prob_index)



    explainer = LimeTextExplainer(class_names = class_names)
    exp = explainer.explain_instance(test_dataframe["Test_Features"][idx], c.predict_proba, num_features = 10,labels=[0, 1,2,3,4,5])
    #exp = explainer.explain_instance(filtered_df["text"][idx], c.predict_proba, num_features = 10)



    print ("True class:",  class_names[int(test_dataframe.True_Labels[idx])])
    print("Probability (pants-fire) =", c.predict_proba([test_dataframe["Test_Features"][idx]])[0, 0])
    print("Probability (false) =", c.predict_proba([test_dataframe["Test_Features"][idx]])[0, 1])
    print("Probability (barely-true) =", c.predict_proba([test_dataframe["Test_Features"][idx]])[0, 2])
    print("Probability (half-true) =", c.predict_proba([test_dataframe["Test_Features"][idx]])[0, 3])
    print("Probability (mostly-true) =", c.predict_proba([test_dataframe["Test_Features"][idx]])[0, 4])
    print("Probability (true) =", c.predict_proba([test_dataframe["Test_Features"][idx]])[0, 5])


  # print(exp.as_list())
  # exp.show_in_notebook(text=filtered_df["text"][idx], labels=[0,1])
  exp.show_in_notebook(text=True)
  exp.save_to_file(f'02-Paraphrasing/Figures/{dataset_name}_{paraphraser}_{feature_name}{model}_sample{idx}.html')
  #exp.save_to_file('kaggle_tf-idf_svm_1.pdf')
  #exp.as_pyplot_figure()


  # exp.as_list()
  # # Try to show explanations for all labels or a subset of valid labels
  # for label in range(len(class_names)):
  #     print(label)
  #     try:
  #         print(label)
  #         exp.show_in_notebook(text=filtered_df["text"][idx], labels=(label,))
  #         print(f"Explanation for label {class_names[label]} shown.")
  #     except KeyError:
  #         print(f"No explanation available for label {class_names[label]}.")





#%%

'''
BERT Explainer

'''
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer




idx = 556
dataset_name = 'covid-19'
paraphraser = 'llama'
feature_name = 'bert'

model = DistilBertForSequenceClassification.from_pretrained(f"./02-Paraphrasing/Results/LLMs/{dataset_name}_{paraphraser}_{feature_name}/")
print(model)



tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
device = torch.device("cuda")
device = torch.device("mps")

test_dataframe = pd.read_excel(f"02-Paraphrasing/Results/LLMs/{dataset_name}_{paraphraser}_{feature_name}_classification_results.xlsx")


#test_text = 'Models forecasting COVID-19 fatalities are referring to scenarios where no preventative measures are taken.'
#test_text = 'model forecasting covid fatality referring scenario preventative measure taken'
test_text=test_dataframe.Test_Features[idx]

def predictor(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    tensor_logits = outputs[0]
    probas = F.softmax(tensor_logits).detach().numpy()
    return probas

if dataset_name == 'covid-19':
    class_names = ["Fake", "True"]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(test_text, predictor, num_features=10)
    exp.show_in_notebook(text=test_text)
    exp.save_to_file(f'02-Paraphrasing/Figures/{dataset_name}_{paraphraser}_BERT_sample_{idx}.html')
    
elif dataset_name == 'liar_6' or dataset_name == 'liar_2':
    class_names = ["pants-fire", "false","barely-true","half-true","mostly-true","true"]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(test_text, predictor, num_features=10, labels=[0, 1,2,3,4,5])
    exp.show_in_notebook(text=test_text)
    exp.save_to_file(f'02-Paraphrasing/Figures/{dataset_name}_{paraphraser}_BERT_sample_{idx}_2.html')
    

print(tokenizer(test_text,return_tensors='pt', padding=True))




#%%

'''
Analysis of sentiment shift
'''

print("Analysis of sentiment shift")

















