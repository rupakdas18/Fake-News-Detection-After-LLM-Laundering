#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os
import matplotlib as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import torch.nn.functional as F
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize



def data_load(dataset_name,class_number,paraphraser,language,classifier):

   
    if dataset_name == 'kaggle':
        
        train_df = pd.read_csv("Dataset/kaggle/train.csv")
        test_df = pd.read_csv("Dataset/kaggle/test.csv")
        
        df = pd.concat([train_df, test_df], axis =0 )
        df.reset_index(inplace = True)
        # drop the index column
        df.drop(["index"], axis = 1, inplace = True)
        # Check the number of columns containing "null"
        print("Numer of data with NULL values: ", df.isnull().sum())
        df = df.dropna()
        # convert the labels to int
        df['label'] = df['label'].astype(int)
      
      
        #g2 = plt.pie(df["target"].value_counts().values,explode=[0,0],labels=df.target.value_counts().index, autopct='%1.1f%%',colors=['SkyBlue','PeachPuff'])
        g2 = plt.pie(df["label"].value_counts().values,explode=[0,0],labels=['Fake news','True news'], autopct='%1.1f%%',colors=['SkyBlue','PeachPuff'])
        plt.show()
      
        print(df.shape)
        print(print(df.label.value_counts()))
        print(df.isnull().sum())
        df = df.dropna()
      
        valid_df = test_df
      
        return df, train_df, test_df, valid_df
    
    ################################################################################################################################
    elif dataset_name == 'TALLIP':
            
        if classifier != 'gpt':

            label_mapping = {
            'Fake': 0,
            'Legit': 1,
        
        }
            
        else:

            label_mapping = {
            'Fake': 'fake',
            'Legit': 'true',
        
        }
        
        if paraphraser == 'human':
            train_dataframe = []
            for filename in os.listdir(f"Dataset/TALLIP/{language}/train/"):
                train_dataframe.append(pd.read_csv(f"Dataset/TALLIP/{language}/train/{filename}", sep="\t"))

            test_dataframe = []
            for filename in os.listdir(f"Dataset/TALLIP/{language}/test/"):
                test_dataframe.append(pd.read_csv(f"Dataset/TALLIP/{language}/test/{filename}", sep="\t"))

        

        train_data = pd.concat(train_dataframe, axis=0, ignore_index=True)
        test_data = pd.concat(test_dataframe, axis=0, ignore_index=True)

        headers = ['domain', 'topic', 'text', 'label']
        train_data.columns = headers
        test_data.columns = headers
        
            


        train_data['label'] = train_data['label'].map(label_mapping)
        test_data['label'] = test_data['label'].map(label_mapping)

        print(train_data.head())

        merged_df = pd.concat([train_data, test_data], axis=0)

        valid_data = test_data



    ################################################################################################################################
    
    elif dataset_name == 'covid-19':
      
      if classifier != 'gpt':

        label_mapping = {
            'fake': 0,
            'real': 1,
        
        }

      else:
          
        label_mapping = {
            'fake': 'fake',
            'real': 'true',
        
        }
    
      if paraphraser == 'human':
    
          train_data=pd.read_csv("Dataset/covid_19/covid_train.tsv",sep='\t')
          test_data=pd.read_csv("Dataset/covid_19/covid_test_with_label.tsv",sep='\t')
          valid_data=pd.read_csv("Dataset/covid_19/covid_valid.tsv",sep='\t')
        
        
          headers = ['id', 'text', 'label']
          train_data.columns = headers
          test_data.columns = headers
          valid_data.columns = headers
        
        
          merged_df = pd.concat([train_data, test_data, valid_data], axis=0)

          print(merged_df.shape)
        
          label_counts = merged_df['label'].value_counts()
          print(label_counts)
        
          merged_df['label'] = merged_df['label'].map(label_mapping)
          train_data['label'] = train_data['label'].map(label_mapping)
          test_data['label'] = test_data['label'].map(label_mapping)
          valid_data['label'] = valid_data['label'].map(label_mapping)
          merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
          
      elif paraphraser == 'llama':
          
          train_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_llama_paraphrased_train.csv")
          test_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_llama_paraphrased_test.csv")
          valid_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_llama_paraphrased_valid.csv")
    
          print(train_data.columns)
          print(train_data.head())
    
          train_data = train_data.drop(['text'], axis=1)
          test_data = test_data.drop(['text'], axis=1)
          valid_data = valid_data.drop(['text'], axis=1)
    
    
          train_data = train_data.rename(columns={"llama": "text"})
          test_data = test_data.rename(columns={"llama": "text"})
          valid_data = valid_data.rename(columns={"llama": "text"})
          
          print(train_data.columns)
          print(train_data.head())
    
          merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
          
          
      elif paraphraser == 'gpt':
          
          train_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_GPT_paraphrased_train.csv")
          test_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_GPT_paraphrased_test.csv")
          valid_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_GPT_paraphrased_valid.csv")
    
          print(train_data.columns)
          print(train_data.head())
    
          train_data = train_data.drop(['text'], axis=1)
          test_data = test_data.drop(['text'], axis=1)
          valid_data = valid_data.drop(['text'], axis=1)
    
          print(train_data.columns)
          print(train_data.head())
    
    
          train_data = train_data.rename(columns={"GPT": "text"})
          test_data = test_data.rename(columns={"GPT": "text"})
          valid_data = valid_data.rename(columns={"GPT": "text"})
    
          merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
          
      elif paraphraser == 'gpt_sentiment':
          
          print("I am here")
          
          df = pd.read_csv('Sentiment_analysis/covid_19_GPT_sentiment.csv')
          
          train_data = df.iloc[0:6418]
          test_data = df.iloc[6419:8536]
          valid_data = df.iloc[8537:]
          
          
          
          train_data = train_data[(train_data['sentiment_shift'] >= -0.4) & (train_data['sentiment_shift'] <= 0.4)]
          test_data = test_data[(test_data['sentiment_shift'] >= -0.5) & (test_data['sentiment_shift'] <= 0.5)]
          valid_data = valid_data[(valid_data['sentiment_shift'] >= -0.5) & (valid_data['sentiment_shift'] <= 0.5)]
          
          train_data = train_data[['text','GPT','label']]
          test_data = test_data[['text','GPT','label']]
          valid_data = valid_data[['text','GPT','label']]

          print(train_data.columns)
          print(train_data.head())
      
          train_data = train_data.drop(['text'], axis=1)
          test_data = test_data.drop(['text'], axis=1)
          valid_data = valid_data.drop(['text'], axis=1)
      
          print(train_data.columns)
          print(train_data.head())
      
      
          train_data = train_data.rename(columns={"GPT": "text"})
          test_data = test_data.rename(columns={"GPT": "text"})
          valid_data = valid_data.rename(columns={"GPT": "text"})
      
          merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
          
          
      elif paraphraser == 'pegasus':
        
          train_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_pegasus_paraphrased_train.csv")
          test_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_pegasus_paraphrased_test.csv")
          valid_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_pegasus_paraphrased_valid.csv")
        
          train_data = train_data.drop(['text'], axis=1)
          test_data = test_data.drop(['text'], axis=1)
          valid_data = valid_data.drop(['text'], axis=1)
        
          train_data = train_data.rename(columns={'pegasus1': 'text'})
          test_data = test_data.rename(columns={'pegasus1': 'text'})
          valid_data = valid_data.rename(columns={'pegasus1': 'text'})
        
          merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
          
      elif paraphraser == 'parrot':
          
            train_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_parrot_paraphrased_train.csv")
            test_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_parrot_paraphrased_test.csv")
            valid_data=pd.read_csv("02-Paraphrasing/covid_19/covid-19_parrot_paraphrased_valid.csv")
          
            train_data = train_data.drop(['text'], axis=1)
            test_data = test_data.drop(['text'], axis=1)
            valid_data = valid_data.drop(['text'], axis=1)
          
            train_data = train_data.rename(columns={'t5_0': 'text'})
            test_data = test_data.rename(columns={'t5_0': 'text'})
            valid_data = valid_data.rename(columns={'t5_0': 'text'})
          
            merged_df = pd.concat([train_data, test_data, valid_data], axis=0)

       
    

  
    
    elif dataset_name == 'liar_2' or dataset_name == 'liar_6':
        
        
        if paraphraser == 'human':
            train_data=pd.read_csv("Dataset/liar/train.tsv",sep='\t')
            test_data=pd.read_csv("Dataset/liar/test.tsv",sep='\t')
            valid_data=pd.read_csv("Dataset/liar/valid.tsv",sep='\t')
            
            train_data = train_data.iloc[:, :3]
            test_data = test_data.iloc[:, :3]
            valid_data = valid_data.iloc[:, :3]
            headers = ['id', 'label', 'text']
            train_data.columns = headers
            test_data.columns = headers
            valid_data.columns = headers

            if classifier != 'gpt':
            
                if class_number == 6:
                    label_mapping = {
                    'pants-fire': 0,
                    'false': 1,
                    'barely-true': 2,
                    'half-true': 3,
                    'mostly-true': 4,
                    'true': 5}
                    
                elif class_number == 2:
                    label_mapping = {
                'pants-fire': 0,
                'false': 0,
                'barely-true': 0,
                'half-true': 1,
                'mostly-true': 1,
                'true': 1
                }
                    
            else:
                if class_number == 6:
                    label_mapping = {
                    'pants-fire': 'pants-fire',
                    'false': 'false',
                    'barely-true': 'barely-true',
                    'half-true': 'half-true',
                    'mostly-true': 'mostly-true',
                    'true': 'true'}
                    
                elif class_number == 2:
                    label_mapping = {
                'pants-fire': 'false',
                'false': 'false',
                'barely-true': 'false',
                'half-true': 'true',
                'mostly-true': 'true',
                'true': 'true'
                }
                
            merged_df = pd.concat([train_data, test_data, valid_data], axis=0)

            print(merged_df.shape)
                
            merged_df['label'] = merged_df['label'].map(label_mapping)
            train_data['label'] = train_data['label'].map(label_mapping)
            test_data['label'] = test_data['label'].map(label_mapping)
            valid_data['label'] = valid_data['label'].map(label_mapping)
            
            
        elif paraphraser == 'pegasus':
            
            train_data=pd.read_csv("02-Paraphrasing/liar/liar_pegasus_paraphrased_train.csv")
            test_data=pd.read_csv("02-Paraphrasing/liar/liar_pegasus_paraphrased_test.csv")
            valid_data=pd.read_csv("02-Paraphrasing/liar/liar_pegasus_paraphrased_valid.csv")
            
            train_data = train_data.drop(['text'], axis=1)
            test_data = test_data.drop(['text'], axis=1)
            valid_data = valid_data.drop(['text'], axis=1)
            
            train_data = train_data.rename(columns={'pegasus1': 'text'})
            test_data = test_data.rename(columns={'pegasus1': 'text'})
            valid_data = valid_data.rename(columns={'pegasus1': 'text'})
            
            merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
            
            if class_number == 2:
                label_mapping = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 1
                    }
                
                merged_df['label'] = merged_df['label'].map(label_mapping)
                train_data['label'] = train_data['label'].map(label_mapping)
                test_data['label'] = test_data['label'].map(label_mapping)
                valid_data['label'] = valid_data['label'].map(label_mapping)
                
                
        elif paraphraser == 'llama':
            
            train_data=pd.read_csv("02-Paraphrasing/liar/liar_llama_paraphrased_train.csv")
            test_data=pd.read_csv("02-Paraphrasing/liar/liar_llama_paraphrased_test.csv")
            valid_data=pd.read_csv("02-Paraphrasing/liar/liar_llama_paraphrased_valid.csv")
      
            print(train_data.columns)
            print(train_data.head())
      
            train_data = train_data.drop(['text'], axis=1)
            test_data = test_data.drop(['text'], axis=1)
            valid_data = valid_data.drop(['text'], axis=1)
      
            print(train_data.columns)
            print(train_data.head())
      
      
            train_data = train_data.rename(columns={"llama": "text"})
            test_data = test_data.rename(columns={"llama": "text"})
            valid_data = valid_data.rename(columns={"llama": "text"})
      
            merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
            
            if class_number == 2:
                label_mapping = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 1
                    }
                
                merged_df['label'] = merged_df['label'].map(label_mapping)
                train_data['label'] = train_data['label'].map(label_mapping)
                test_data['label'] = test_data['label'].map(label_mapping)
                valid_data['label'] = valid_data['label'].map(label_mapping)
            
            
            
            
            
        elif paraphraser == 'parrot':       
            
            

            train_data=pd.read_csv("02-Paraphrasing/liar/parrot_paraphrased_train.csv")
            test_data=pd.read_csv("02-Paraphrasing/liar/parrot_paraphrased_test.csv")
            valid_data=pd.read_csv("02-Paraphrasing/liar/parrot_paraphrased_valid.csv")
        
            train_data = train_data.drop(['text','t5_1','t5_2','Unnamed: 0'], axis=1)
            test_data = test_data.drop(['text','t5_1','t5_2','Unnamed: 0'], axis=1)
            valid_data = valid_data.drop(['text','t5_1','t5_2','Unnamed: 0'], axis=1)
            
            train_data = train_data.rename(columns={'t5_0': 'text'})
            test_data = test_data.rename(columns={'t5_0': 'text'})
            valid_data = valid_data.rename(columns={'t5_0': 'text'})
            
            merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
            
            if class_number == 2:
                label_mapping = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 1
                    }
                
                merged_df['label'] = merged_df['label'].map(label_mapping)
                train_data['label'] = train_data['label'].map(label_mapping)
                test_data['label'] = test_data['label'].map(label_mapping)
                valid_data['label'] = valid_data['label'].map(label_mapping)
            
            
        elif paraphraser == 'bard':
            train_data=pd.read_csv("02-Paraphrasing/liar/bard_paraphrased_train.csv")
            test_data=pd.read_csv("02-Paraphrasing/liar/bard_paraphrased_test.csv")
            valid_data=pd.read_csv("02-Paraphrasing/liar/bard_paraphrased_valid.csv")
            
            train_data = train_data.drop(['text','Unnamed: 0'], axis=1)
            test_data = test_data.drop(['text','Unnamed: 0'], axis=1)
            valid_data = valid_data.drop(['text','Unnamed: 0'], axis=1)
            
            train_data = train_data.rename(columns={'bart': 'text'})  # need to edit the name  of the paraphraser
            test_data = test_data.rename(columns={'bart': 'text'})
            valid_data = valid_data.rename(columns={'bart': 'text'})
            
            merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
            
            
            if class_number == 2:
                label_mapping = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 1
                    }
                
                merged_df['label'] = merged_df['label'].map(label_mapping)
                train_data['label'] = train_data['label'].map(label_mapping)
                test_data['label'] = test_data['label'].map(label_mapping)
                valid_data['label'] = valid_data['label'].map(label_mapping)
                
                
        elif paraphraser == 'gpt':
            
            train_data=pd.read_csv("02-Paraphrasing/liar/liar_GPT_paraphrased_train.csv")
            test_data=pd.read_csv("02-Paraphrasing/liar/liar_GPT_paraphrased_test.csv")
            valid_data=pd.read_csv("02-Paraphrasing/liar/liar_GPT_paraphrased_valid.csv")
      
            print(train_data.columns)
            print(train_data.head())
      
            train_data = train_data.drop(['text'], axis=1)
            test_data = test_data.drop(['text'], axis=1)
            valid_data = valid_data.drop(['text'], axis=1)
      
            print(train_data.columns)
            print(train_data.head())
      
      
            train_data = train_data.rename(columns={"GPT": "text"})
            test_data = test_data.rename(columns={"GPT": "text"})
            valid_data = valid_data.rename(columns={"GPT": "text"})
      
            merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
            
            if class_number == 2:
                label_mapping = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 1
                    }
                
                merged_df['label'] = merged_df['label'].map(label_mapping)
                train_data['label'] = train_data['label'].map(label_mapping)
                test_data['label'] = test_data['label'].map(label_mapping)
                valid_data['label'] = valid_data['label'].map(label_mapping)
            
        
    #merged_df = pd.concat([train_data, test_data, valid_data], axis=0)
    label_counts = merged_df['label'].value_counts()
    print(label_counts)

    #g2 = plt.pie(merged_df["label"].value_counts().values,explode=[0,0,0,0,0,0],labels=['half-true','false','mostly-true','barely-true','true','pants-fire'], autopct='%1.1f%%')
    #g2 = plt.pie(merged_df["label"].value_counts().values,explode=[0,0,0,0,0,0],labels=merged_df.label.value_counts().index, autopct='%1.1f%%',colors=['SkyBlue','PeachPuff'])
    plt.show()
    
    print("Shape before removing NAN values")
    print(train_data.shape)
    print(test_data.shape)
    print(valid_data.shape)


    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')
    valid_data = valid_data.dropna(how='any')

    print("Shape after removing NAN values")

    print(train_data.shape)
    print(test_data.shape)
    print(valid_data.shape)

              
    return merged_df, train_data, test_data, valid_data
    
    

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
