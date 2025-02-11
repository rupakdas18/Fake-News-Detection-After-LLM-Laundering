# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:00:58 2024

@author: rjd6099

Text classification using T5 model

"""



from datasets import Dataset, DatasetDict
import load_data

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
metric = evaluate.load("f1")
from sklearn.metrics import (accuracy_score,
                                   classification_report,
                                   confusion_matrix)
      
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score




def data_processing(train_df,test_df,valid_df):
    
    
  train_df['target'] = train_df['target'].astype(str)
  test_df['target'] = test_df['target'].astype(str)
  valid_df['target'] = valid_df['target'].astype(str)


  # Converting pandas DataFrames into Hugging Face Dataset objects:
  dataset_train = Dataset.from_pandas(train_df.drop('label',axis=1))
  dataset_val = Dataset.from_pandas(valid_df.drop('label',axis=1))
  dataset_test = Dataset.from_pandas(test_df.drop('label',axis=1))
# Shuffle the training dataset
  dataset_train_shuffled = dataset_train.shuffle(seed=42)  # Using a seed for reproducibility
  
 

# Combine them into a single DatasetDict
  dataset = DatasetDict({
      'train': dataset_train_shuffled,
      'val': dataset_val,
      'test': dataset_test
  })
  
  print(dataset)

  return dataset


def label_convert(df):
  df['label']=df['label'].astype('category')
  df['target']=df['label'].cat.codes

  return df



def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = [item for item in sample["text"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["target"], max_length=max_target_length, padding=padding, truncation=True)
    
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result



def printt_classification_report(str_labels_list,predictions_list):
         
      
    print("Confusion Matrix:")
    print(confusion_matrix(str_labels_list, predictions_list))
      
    print("\nClassification Report:")
    print(classification_report(str_labels_list, predictions_list))
      
    print("Accuracy Score:", accuracy_score(str_labels_list, predictions_list))
      
    f1 = f1_score(str_labels_list, predictions_list, average='weighted')
    precision = precision_score(str_labels_list, predictions_list, average='weighted')
    recall = recall_score(str_labels_list, predictions_list, average='weighted')
      
    print(f"F1 Score: {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    
    
    
if __name__ == "__main__":
    
    
  dataset_name = 'covid-19' # 'liar_2', 'liar_6', 'kaggle', covid-19
  num_labels = 2 # 2, 6
  paraphraser = 'parrot' # 'no', 'bard', 'parrot','gpt', 'pegasus','llama'
    
    

    
  full_df, train_df, test_df, valid_df = load_data.load_data(dataset_name,num_labels,paraphraser)

  text_length = 4500
  train_df['clean_text']=train_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
  test_df['clean_text']=test_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
  valid_df['clean_text']=valid_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
  
  
  print(train_df.head())
  #train_data,test_data,eval_data = data_processing(train_df['clean_text'],test_df['clean_text'],valid_df['clean_text'])

  train_df = label_convert(train_df)
  test_df = label_convert(test_df)
  valid_df = label_convert(valid_df)

  category_map = {code: category for code, category in enumerate(train_df['label'].cat.categories)}
  print(category_map)

  print(train_df.head())

  dataset = data_processing(train_df,test_df,valid_df)
  print(dataset)
  
  print(dataset['train'][1])
  

  model_id="google/flan-t5-base"

# Load tokenizer of FLAN-t5-base
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  
  
# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
  tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["clean_text"], truncation=True), batched=True, remove_columns=['clean_text', 'target'])
  max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
  print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
  tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["target"], truncation=True), batched=True, remove_columns=['clean_text', 'target'])
  max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
  print(f"Max target length: {max_target_length}")

  
  
  tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['clean_text', 'target'])
  print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
  
  
  from transformers import AutoModelForSeq2SeqLM

  # huggingface hub model id
  model_id="google/flan-t5-base"
    
  # load model from the hub
  model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
  
  
  from transformers import DataCollatorForSeq2Seq

    # we want to ignore tokenizer pad token in the loss
  label_pad_token_id = -100
    # Data collator
  data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
)
  
  from huggingface_hub import login, hf_hub_download

  login(token='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')
  
  #from huggingface_hub import HfFolder
  from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hugging Face repository id
  #repository_id = f"{model_id.split('/')[1]}-imdb-text-classification"
  repository_id = 'results'

# Define training args
  training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=3e-4,
    num_train_epochs=2,
    # logging & evaluation strategies
    #logging_dir=f"{repository_id}/logs",
    #logging_strategy="epoch", 
    # logging_steps=1000,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    #report_to="tensorboard",
    #push_to_hub=True,
    #hub_strategy="every_save",
    #hub_model_id=repository_id,
    #hub_token=HfFolder.get_token(),
)

# Create Trainer instance
  trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    compute_metrics=compute_metrics,
)


  trainer.train()
    
  trainer.evaluate()
    
    
    
  from tqdm.auto import tqdm
    
  samples_number = len(dataset['test'])
  progress_bar = tqdm(range(samples_number))
  predictions_list = []
  labels_list = []
  for i in range(samples_number):
      text = dataset['test']['text'][i]
      inputs = tokenizer.encode_plus(text, padding='max_length', max_length=512, return_tensors='pt').to('cuda')
      outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
      prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
      predictions_list.append(prediction)
      labels_list.append(dataset['test']['target'][i])
    
      progress_bar.update(1)
      
  str_labels_list = []
  for i in range(len(labels_list)): str_labels_list.append(str(labels_list[i]))
  
  printt_classification_report(str_labels_list,predictions_list)


    
    
    


