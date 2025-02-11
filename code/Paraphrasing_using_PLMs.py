# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:35:18 2024

@author: rjd6099
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import torch
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from dotenv import load_dotenv
load_dotenv()



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




#%%

"""
Data Processing and this is for liar dataset

"""

import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd


def load_data_liar():

  label_mapping = {
    'pants-fire': 0,
    'false': 1,
    'barely-true': 2,
    'half-true': 3,
    'mostly-true': 4,
    'true': 5
}


  train_data=pd.read_csv("liar/train.tsv",sep='\t')
  test_data=pd.read_csv("liar/test.tsv",sep='\t')
  valid_data=pd.read_csv("liar/valid.tsv",sep='\t')

  train_data = train_data.iloc[:, :3]
  test_data = test_data.iloc[:, :3]
  valid_data = valid_data.iloc[:, :3]
  
  headers = ['id', 'label', 'text']
  
  train_data.columns = headers
  test_data.columns = headers
  valid_data.columns = headers




  train_data['label'] = train_data['label'].map(label_mapping)
  test_data['label'] = test_data['label'].map(label_mapping)
  valid_data['label'] = valid_data['label'].map(label_mapping)
  
  
  print("Shape of training dataset",train_data.shape)
  print("Shape of testing dataset",test_data.shape)
  print("Shape of validating dataset",valid_data.shape)


  return train_data, test_data, valid_data




#%%%

"""
Data Processing and this is for kaggle dataset

"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch



def load_data_kaggle():


  train_df=pd.read_csv("kaggle/train.csv")



  train_df.reset_index(inplace = True)

  # drop the index column
  train_df.drop(["index"], axis = 1, inplace = True)

  # Check the number of columns containing "null"
  print("Numer of data with NULL values: ", train_df.isnull().sum())
  train_df = train_df.dropna()

  # convert the labels to int
  train_df['label'] = train_df['label'].astype(int)
  
  
  return train_df


dataframe = load_data_kaggle()


text_length = 1024
dataframe['clean_text']=dataframe['text'].apply(lambda x: clean_text(x,text_length))
print(dataframe.shape)
print(dataframe)

#%%

"""
Data Processing and this is for covid_19 dataset

"""


import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd

def covid_19_data_load():

  label_mapping = {
      'fake': 0,
      'real': 1,

  }

  train_data=pd.read_csv("covid_19/covid_train.tsv",sep='\t')
  valid_data=pd.read_csv("covid_19/covid_valid.tsv",sep='\t')
  test_data=pd.read_csv("covid_19/covid_test_with_label.tsv",sep='\t')



  headers = ['id', 'text', 'label']
  train_data.columns = headers
  valid_data.columns = headers
  test_data.columns = headers



  train_data['label'] = train_data['label'].map(label_mapping)
  valid_data['label'] = valid_data['label'].map(label_mapping)
  test_data['label'] = test_data['label'].map(label_mapping)

  print("Shape of training data: ",train_data.shape)
  print("Shape of validation data: ", valid_data.shape)
  print("Shape of testing data: ", test_data.shape)
  print("--------------------------------------------------------------------")

  return train_data, test_data, valid_data




#%%
   
"""
T5 Paraphraser (Parrot)
Source code: //github.com/PrithivirajDamodaran/Parrot_Paraphraser.git

"""
from parrot import Parrot


# Function to set a random seed for reproducibility
def random_state(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

random_state(1234)

# Initialize the Parrot model (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")



# # Function to generate the best paraphrase for a given input sentence
# def get_paraphrase(phrase):
#     para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False, max_length=512)
#     if para_phrases:
#         # You can modify this to select the best paraphrase based on your criteria
#         # For simplicity, I'm choosing the first paraphrase in the list
#         return para_phrases[1][0]
#     else:
#         return phrase  # Return the original phrase if no paraphrases are generated

# # Apply the paraphrase function to each row in the 'text' column and save to a new column 'T5'
# dataframe['T5'] = dataframe['text'].apply(get_paraphrase)

# # Display the DataFrame with the original and paraphrased sentences



for i, reference in dataframe["text"].items():
  # if dataframe.loc[i, "t5_0"] != '':
  #   continue
  # else:
    try:
      responses = parrot.augment(input_phrase=reference, 
                                use_gpu=True,
                                do_diverse=False,
                                diversity_ranker="levenshtein",
                                adequacy_threshold = 0.5,
                                fluency_threshold = 0.5
                                )
      for j, response in enumerate(responses):
        column_name = f"t5_{j}"
        dataframe.loc[i, column_name] = response[0]
        # print(f"{i}-{column_name} = {response[0]}")
        if j == 2:
          break
    except:
      print(f"{i} - No paraphrases returned")



print(dataframe)
dataframe.to_csv("covid_9_parrot_paraphrased_test.csv")
   
   
#%%

"""
T5 Paraphraser (Pegasus)
Source code: https://github.com/google-research/pegasus

"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

num_beams = 3
num_return_sequences = 3

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  #print(batch['input_ids'].shape)
  num_tokens = batch['input_ids'].shape[1]
  if num_tokens > 60:
      return "The length is too long"
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


# context = "Last note: Washington DC's total test count fell by ~22% presumably pulling out antibody tests."
# get_response(context,num_return_sequences,num_beams)


#dataframe[["pegasus1",'pegasus2','pegasus3']] = dataframe.apply(lambda row: get_response(row['clean_text'],num_beams,num_return_sequences), axis=1,result_type="expand")
dataframe[["pegasus1",'pegasus2','pegasus3']] = dataframe.apply(lambda row: get_response(row['text'],num_beams,num_return_sequences), axis=1,result_type="expand")

dataframe.to_csv("liar_pegasus_paraphrased_valid.csv")

#%%
"""
Replicate paraphraser using API (Not free)

"""
import os
import replicate

api_key = os.getenv('REPLICATE_API_TOKEN')


def Llama_paraphrase(prompt_input):
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                            input={"prompt": f"Paraphrase the following text without writing any additional comment: {prompt_input} ", # Prompts
                            "temperature":0.1, "top_p":0.9, "max_length":128, "repetition_penalty":1})  # Model parameters

    
    
    output_lines = ""
    
    
    for item in output:
      output_lines += item
      
    lines = output_lines.splitlines()
    Llama_paraphrased = " ".join(lines[1:]).strip()
    
    #print(Llama_paraphrased)
      

      
    return Llama_paraphrased


dataframe["llama/replicate"] = dataframe.apply(lambda row: Llama_paraphrase(row['text']), axis=1)
dataframe.to_csv("replicate_paraphrased.csv")





#%%

"""
Bart model

"""


import torch
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model = model.to(device)

def generate_response(input_sentence, model, device):
    tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
    batch = tokenizer(input_sentence, return_tensors='pt')
    
    # Check if the number of tokens exceeds 1024
    num_tokens = batch['input_ids'].shape[1]
    if num_tokens > 1024:
        return "The length is too long"
    
    # Move input tensors to the same device as the model
    batch = {k: v.to(device) for k, v in batch.items()}
    
    generated_ids = model.generate(batch['input_ids'])
    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_sentence[0]

# Apply the function to the dataframe and save the output
dataframe["bart"] = dataframe.apply(lambda row: generate_response(row['text'], model, device), axis=1)
dataframe.to_csv("covid_19_bart_paraphrased_train_2.csv")



#%%

# Modified BART model

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model = model.to(device)

tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')



for i, input_sentence in dataframe["text"].items():
  # if dataframe.loc[i, "t5_0"] != '':
  #   continue
  # else:
    try:
        batch = tokenizer(input_sentence, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        generated_ids = model.generate(batch['input_ids'])
        generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        dataframe.loc[i, 'Bard'] = generated_sentence[0]
        
        
    except:
      print(f"{i} - No paraphrases returned")



print(dataframe)
dataframe.to_csv("covid_9_bard_paraphrased_train.csv")

#%%


# Modified BART model


import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Loop through the dataframe
for i, input_sentence in dataframe["text"].items():
    try:
        # Tokenize the input sentence and check the number of tokens
        inputs = tokenizer(input_sentence, return_tensors='pt')
        num_tokens = inputs['input_ids'].shape[1]

        if num_tokens > 1024:
            # If the number of tokens exceeds 1024, add a warning message to the dataframe
            dataframe.loc[i, 'Bard'] = "The length is too long"
        else:
            # Process the sentence for paraphrasing
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_ids = model.generate(inputs['input_ids'])
            generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            dataframe.loc[i, 'Bard'] = generated_sentence[0]

    except Exception as e:
        print(f"{i} - No paraphrases returned. Error: {e}")

# Save the resulting dataframe
print(dataframe)
dataframe.to_csv("covid_9_bard_paraphrased_train.csv")

#%%


# Modified BART model


import torch
from transformers import BartForConditionalGeneration, BartTokenizer

input_sentence = "COVID-19 Is Caused By A Bacterium, Not Virus And Can Be Treated With Aspirin"

model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
batch = tokenizer(input_sentence, return_tensors='pt')
batch = {k: v.to(device) for k, v in batch.items()}
generated_ids = model.generate(batch['input_ids'])
generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_sentence)



#%%


''' 
# CHATGPT API (NOT FREE)

'''


import openai
import os


# Set up your OpenAI API key
openai.api_key = os.environ.get('openai_api_key')

def paraphrase_text(text):
    # Call the OpenAI API with the input text
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",  # Use "gpt-4" if you have access to it
      messages=[
        {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
        {"role": "user", "content": f"Paraphrase the following text: {text}"}
      ],
      max_tokens=150,  # Adjust the number of tokens if you need a longer or shorter paraphrase
      temperature=0.7,  # Controls the creativity of the output
    )

    # Extract the paraphrased text from the API response
    paraphrased_text = response['choices'][0]['message']['content']
    return paraphrased_text

train_df, test_df, valid_df = load_data_liar() 



dataframe = valid_df
dataframe['GPT'] = dataframe['text'].apply(paraphrase_text)
#dataframe.to_csv('covid_19_GPT_paraphrased_test.csv')
dataframe.to_csv('test_GPT.csv')

print(dataframe)

#%%

'''
LLAMA API (NOT FREE)

''' 

from llamaapi import LlamaAPI
import json
import os
import pandas as pd



api_key = os.environ.get('llama_api_key')
llama=LlamaAPI(api_key)





# API Request JSON Cell
def llama_paraphraser(text,max_tokens=150, temperature=0.7):
  api_request_json = {
    "model": "llama-70b-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
      {"role": "user", "content": f"Paraphrase the following text without any additional comments: {text}."},
    ],
    "max_tokens": max_tokens,
    "temperature": temperature,
  }

  # Run Llama
  try:
      response = llama.run(api_request_json)
      response_data = response.json()

      paraphrased_text = response_data['choices'][0]['message']['content']
      return paraphrased_text.strip()
  except Exception as e:
      print(f"Error during API request: {e}")
      

#train_df, test_df, valid_df = covid_19_data_load() 
#train_df, test_df, valid_df = load_data_liar() 

dataframe = pd.read_excel("lamma_remaining.xlsx")
print(dataframe)


#dataframe = train_df ### Change this line
dataframe['llama'] = dataframe['text'].apply(llama_paraphraser)
dataframe.to_csv('LIAR_llama_paraphrased_train_remaining.csv')




#%%


'''
LLAMA API (NOT FREE) without apply,



''' 

from llamaapi import LlamaAPI
import json
import os
import pandas as pd
import time



api_key = os.environ.get('llama_api_key')
llama = LlamaAPI(api_key)



# API Request JSON Cell
def llama_paraphraser(text, max_tokens=150, temperature=0.7):
    api_request_json = {
        "model": "llama-70b-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
            {"role": "user", "content": f"Paraphrase the following text without any additional comments: {text}."},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Run Llama
    try:
        response = llama.run(api_request_json)
        response_data = response.json()

        paraphrased_text = response_data['choices'][0]['message']['content']
        return paraphrased_text.strip()
    except Exception as e:
        print(f"Error during API request: {e}")
        return None  # In case of error, return None


# # Loading the dataset
# train_df, test_df, valid_df = load_data_liar()

# # Set the dataframe for processing
# dataframe = train_df  # Modify this line for other dataframes if needed

# # Iterate over each row and generate paraphrased text
# paraphrased_texts = []
# for index, row in dataframe.iterrows():
#     text = row['text']
#     paraphrased_text = llama_paraphraser(text)
#     paraphrased_texts.append(paraphrased_text)
#     time.sleep(1)

# # Add paraphrased texts to the dataframe
# dataframe['llama'] = paraphrased_texts

# # Save the updated dataframe to a CSV file
# dataframe.to_csv('liar_llama_paraphrased_train.csv', index=False)



test_text = 'Models forecasting COVID-19 fatalities are referring to scenarios where no preventative measures are taken.'


print(llama_paraphraser(test_text))

   
   
   