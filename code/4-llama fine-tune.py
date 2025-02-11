import time
import load_data

start = time.time()


import warnings
warnings.filterwarnings("ignore")
from accelerate import PartialState, Accelerator
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from peft import AutoPeftModelForCausalLM
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
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from huggingface_hub import login, hf_hub_download

print(torch.cuda.is_available())








def data_processing(train_df,test_df,valid_df):

  # X_train = pd.DataFrame(X_train)
  # X_test = pd.DataFrame(X_test)
  # X_eval = pd.DataFrame(X_eval)

  # train_data = Dataset.from_pandas(X_train)
  # test_data = Dataset.from_pandas(X_test)
  # eval_data = Dataset.from_pandas(X_eval)

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


def define_model(model_name,num_labels):


#   bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float32,
#     bnb_4bit_use_double_quant=True
# )

#   model = AutoModelForCausalLM.from_pretrained(
#       model_name,
#       torch_dtype=torch.float32,
#       quantization_config=bnb_config,
#       device_map = torch.cuda.set_device(desired_device)
#   )

  # model.config.use_cache = False
  # model.config.pretraining_tp = 1
  # tokenizer = AutoTokenizer.from_pretrained(model_name,
  #                                         trust_remote_code=True,
  #                                        )
  # tokenizer.pad_token = tokenizer.eos_token
  # tokenizer.padding_side = "right"
  # model, tokenizer = setup_chat_format(model, tokenizer)

  quantization_config = BitsAndBytesConfig(
      load_in_4bit = True, # enable 4-bit quantization
      bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
      bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
      bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
  )

  lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
  )

  model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels= num_labels
)

  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, lora_config)

  tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

  tokenizer.pad_token_id = tokenizer.eos_token_id
  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.use_cache = False
  model.config.pretraining_tp = 1


  return model,tokenizer


def fine_tune(model, train_data, eval_data, epoch,output_dir):

#   peft_config = LoraConfig(
#         lora_alpha=16,
#         lora_dropout=0.05,
#         r=64,
#         bias="none",
#         target_modules="all-linear",
#         task_type="CAUSAL_LM",
# )

#   training_arguments = TrainingArguments(
#     output_dir=output_dir,                    # directory to save and repository id
#     num_train_epochs=epoch,                       # number of training epochs
#     per_device_train_batch_size=1,            # batch size per device during training
#     gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
#     gradient_checkpointing=True,              # use gradient checkpointing to save memory
#     optim="paged_adamw_32bit",
#     save_steps=0,
#     logging_steps=25,                         # log every 10 steps
#     learning_rate=2e-4,                       # learning rate, based on QLoRA paper
#     weight_decay=0.001,
#     fp16=True,
#     bf16=False,
#     max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
#     max_steps=-1,
#     warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
#     group_by_length=True,
#     lr_scheduler_type="cosine",               # use cosine learning rate scheduler
#     report_to="tensorboard",                  # report metrics to tensorboard
#     evaluation_strategy="epoch"               # save checkpoint every epoch
# )

#   trainer = SFTTrainer(
#       model=model,
#       args=training_arguments,
#       train_dataset=train_data,
#       eval_dataset=eval_data,
#       peft_config=peft_config,
#       dataset_text_field="clean_text",
#       tokenizer=tokenizer,
#       max_seq_length=1024,
#       packing=False,
#       dataset_kwargs={
#           "add_special_tokens": False,
#           "append_concat_token": False,
#       }
#   )

  training_args = TrainingArguments(
    output_dir = 'fake_news_classification',
    learning_rate = 1e-4,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = epoch,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True
)
  trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
)

  return trainer


def label_convert(df):
  df['label']=df['label'].astype('category')
  df['target']=df['label'].cat.codes

  return df


def weight_calculate(df):

  df.target.value_counts(normalize=True)

  class_weights=(1/df.target.value_counts(normalize=True).sort_index()).tolist()
  class_weights=torch.tensor(class_weights)
  class_weights=class_weights/class_weights.sum()

  return class_weights


def get_performance_metrics(df_test):
  y_test = df_test.label
  y_pred = df_test.predictions

  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))

  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
  print("Accuracy Score:", accuracy_score(y_test, y_pred))

  f1 = f1_score(y_test, y_pred, average='weighted')
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')

  print(f"F1 Score: {f1:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")




def llama_preprocessing_function(examples):
    return tokenizer(examples['clean_text'], truncation=True, max_length=MAX_LEN)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),'accuracy':accuracy_score(predictions,labels)}


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    # def compute_loss(self, model, inputs, num_items_in_batch=None):
    #     # Custom loss computation logic here
    #     outputs = model(**inputs)
    #     loss = outputs.loss
    #     return loss

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get('logits')

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

    



def make_predictions(model,df_test,category_map):

  # Convert summaries to a list
  sentences = df_test.clean_text.tolist()

  # Define the batch size
  batch_size = 32  # You can adjust this based on your system's memory capacity

  # Initialize an empty list to store the model outputs
  all_outputs = []

  # Process the sentences in batches
  for i in range(0, len(sentences), batch_size):
      # Get the batch of sentences
      batch_sentences = sentences[i:i + batch_size]

      # Tokenize the batch
      inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

      # Move tensors to the device where the model is (e.g., GPU or CPU)
      inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

      # Perform inference and store the logits
      with torch.no_grad():
          outputs = model(**inputs)
          all_outputs.append(outputs['logits'])


  final_outputs = torch.cat(all_outputs, dim=0)
  df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
  df_test['predictions']=df_test['predictions'].apply(lambda l:category_map[l])

  return df_test


if __name__ == "__main__":
    
    
  dataset_name = 'TALLIP' # 'liar', 'kaggle', covid-19
  num_labels = 6 # 2, 6
  paraphraser = 'human' # 'no', 'bard', 'parrot','gpt', 'pegasus', 'llama''
  language = 'English' # 'english', 'german', 'french',
    

    
  full_df, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_labels,paraphraser,language)

  #list_available_gpus()
  #train_df, test_df, valid_df,num_labels = load_liar2()
  #train_df, test_df, valid_df,num_labels = load_data()

  '''
  This part is just to use a small set of data. Remove that when the code is completed and working

  '''
  
  print("Shape before removing NAN values")
  print(train_df.shape)
  print(test_df.shape)
  print(valid_df.shape)


  train_df = train_df.dropna(how='any')
  test_df = test_df.dropna(how='any')
  valid_df = valid_df.dropna(how='any')

  print("Shape after removing NAN values")

  print(train_df.shape)
  print(test_df.shape)
  print(valid_df.shape)

  # train_df = train_df.sample(n=100)
  # test_df = test_df.sample(n=20)
  # valid_df = train_df.sample(n=20)
  
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

  class_weights = weight_calculate(train_df)





  # model_name = "meta-llama/Llama-2-7b-chat-hf"
  login(token='your api code')
  model_name = "meta-llama/Meta-Llama-3-8B"
  model, tokenizer = define_model(model_name,num_labels)

  test_df = make_predictions(model,test_df,category_map)
  get_performance_metrics(test_df)
  
  mid = time.time()
  print("Time required for classification without fine-tuning: ", (mid-start)/3600)

  MAX_LEN = 512
  # col_to_delete = ['id', 'text']
  col_to_delete = []

  collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)


  tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
  tokenized_datasets = tokenized_datasets.rename_column("target", "label")
  tokenized_datasets.set_format("torch")






  output_dir="trained_weigths_2"
  trainer = fine_tune(model, tokenized_datasets['train'], tokenized_datasets['val'], 5,output_dir)
  train_result = trainer.train()

  make_predictions(model,test_df,category_map)
  get_performance_metrics(test_df)
  
  end = time.time()
  print("Time required for classification with fine-tuning: ", (end-start)/3600)