# -*- coding: utf-8 -*-

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import pandas as pd
import load_data


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

learning_rate = 1e-5
adam_epsilon = 1e-8
num_epochs = 5




def bert_data_prep(X_train, y_train, X_val, y_val, X_test, y_test):
  texts_train = X_train.values
  labels_train = y_train.values
  texts_val = X_val.values
  labels_val = y_val.values
  texts_test = X_test.values
  labels_test = y_test.values
  
  train_text_lengths = [len(texts_train[i].split()) for i in range(len(texts_train))]
  print("Minimum length befor tokenization", min(train_text_lengths))
  print("Maximum length befor tokenization", max(train_text_lengths))
  
  val_text_lengths = [len(texts_val[i].split()) for i in range(len(texts_val))]
  print("Minimum length befor tokenization", min(val_text_lengths))
  print("Maximum length befor tokenization", max(val_text_lengths))

  test_text_lengths = [len(texts_test[i].split()) for i in range(len(texts_test))]
  print("Minimum length befor tokenization", min(test_text_lengths))
  print("Maximum length befor tokenization", max(test_text_lengths))
  
  

  sum([1 for i in range(len(train_text_lengths)) if train_text_lengths[i] >= 300])
  sum([1 for i in range(len(val_text_lengths)) if val_text_lengths[i] >= 300])
  sum([1 for i in range(len(test_text_lengths)) if test_text_lengths[i] >= 300])
  
  
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
  train_text_ids = [tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in texts_train]
  val_text_ids = [tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in texts_val]
  test_text_ids = [tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in texts_test]
  
  

  train_text_ids_lengths = [len(train_text_ids[i]) for i in range(len(train_text_ids))]
  val_text_ids_lengths = [len(val_text_ids[i]) for i in range(len(val_text_ids))]
  test_text_ids_lengths = [len(test_text_ids[i]) for i in range(len(test_text_ids))]
  
  
  print("Minimum length after tokenization", min(train_text_ids_lengths))
  print("Maximum length after tokenization", max(train_text_ids_lengths))
  print("Minimum length after tokenization", min(val_text_ids_lengths))
  print("Maximum length after tokenization", max(val_text_ids_lengths))
  print("Minimum length after tokenization", min(test_text_ids_lengths))
  print("Maximum length after tokenization", max(test_text_ids_lengths))

  train_att_masks = []
  for ids in train_text_ids:
    masks = [int(id > 0) for id in ids]
    train_att_masks.append(masks)
    
  val_att_masks = []
  for ids in val_text_ids:
    masks = [int(id > 0) for id in ids]
    val_att_masks.append(masks)
    
  test_att_masks = []
  for ids in test_text_ids:
    masks = [int(id > 0) for id in ids]
    test_att_masks.append(masks)

  # train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, labels, random_state=111, test_size=0.4)
  # train_m, test_val_m = train_test_split(att_masks, random_state=111, test_size=0.4)

  # test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=111, test_size=0.5)
  # test_m, val_m = train_test_split(test_val_m, random_state=111, test_size=0.5)

  train_x = torch.tensor(train_text_ids)
  test_x = torch.tensor(test_text_ids)
  val_x = torch.tensor(val_text_ids)
  train_y = torch.tensor(labels_train)
  test_y = torch.tensor(labels_test)
  val_y = torch.tensor(labels_val)
  train_m = torch.tensor(train_att_masks)
  test_m = torch.tensor(test_att_masks)
  val_m = torch.tensor(val_att_masks)

  batch_size = 32

  train_data = TensorDataset(train_x, train_m, train_y)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  val_data = TensorDataset(val_x, val_m, val_y)
  val_sampler = SequentialSampler(val_data)
  val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

  test_data = TensorDataset(test_x, test_m)
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

  num_labels = len(set(labels_train))
  model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  model = model.to(device)

  parameter_number = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('Number of trainable parameters:', parameter_number, '\n', model)

  return model, train_dataloader, val_dataloader, test_dataloader, device, tokenizer, test_y


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




def bert_optimizer(model, learning_rate, adam_epsilon):

  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.2},
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.0}
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
  return optimizer


from transformers import get_linear_schedule_with_warmup
def bert_scheduler(num_epochs, train_dataloader, optimizer):

  total_steps = len(train_dataloader) * num_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  return scheduler



def train_bert_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device):
  train_losses = []
  val_losses = []
  num_mb_train = len(train_dataloader)
  num_mb_val = len(val_dataloader)

  if num_mb_val == 0:
      num_mb_val = 1

  for n in range(num_epochs):
      train_loss = 0
      val_loss = 0
      start_time = time.time()

      for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):
          optimizer.zero_grad()
          model.train()

          mb_x = mb_x.to(device)
          mb_m = mb_m.to(device)
          mb_y = mb_y.to(device)

          outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)

          loss = outputs[0]
          #loss = model_loss(outputs[1], mb_y)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()

          train_loss += loss.data / num_mb_train

      print ("\nTrain loss after itaration %i: %f" % (n+1, train_loss))
      train_losses.append(train_loss.cpu())

      with torch.no_grad():
          model.eval()

          for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
              mb_x = mb_x.to(device)
              mb_m = mb_m.to(device)
              mb_y = mb_y.to(device)

              outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)

              loss = outputs[0]
              #loss = model_loss(outputs[1], mb_y)

              val_loss += loss.data / num_mb_val

          print ("Validation loss after itaration %i: %f" % (n+1, val_loss))
          val_losses.append(val_loss.cpu())

      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      print(f'Time: {epoch_mins}m {epoch_secs}s')


  return model, train_losses, val_losses


import pickle

def save_bert_model(model, path, tokenizer,train_losses,val_losses):

  model_to_save = model.module if hasattr(model, 'module') else model
  model_to_save.save_pretrained(path)
  tokenizer.save_pretrained(path)

  with open(path + '/train_losses.pkl', 'wb') as f:
      pickle.dump(train_losses, f)

  with open(path + '/val_losses.pkl', 'wb') as f:
      pickle.dump(val_losses, f)

  model = DistilBertForSequenceClassification.from_pretrained(path)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)

  with open(path + '/train_losses.pkl', 'rb') as f:
      train_losses = pickle.load(f)

  with open(path + '/val_losses.pkl', 'rb') as f:
      val_losses = pickle.load(f)

  plt.figure()
  plt.plot(train_losses)
  plt.figure()
  plt.plot(val_losses)
  plt.show()
  
  
  

def test_bert_model(model, test_dataloader, device, test_y):
  outputs = []
  with torch.no_grad():
      model.eval()
      for k, (mb_x, mb_m) in enumerate(test_dataloader):
          mb_x = mb_x.to(device)
          mb_m = mb_m.to(device)
          output = model(mb_x, attention_mask=mb_m)
          outputs.append(output[0].to('cpu'))

  outputs = torch.cat(outputs)
  _, predicted_values = torch.max(outputs, 1)
  predicted_values = predicted_values.numpy()
  true_values = test_y.numpy()
  test_accuracy = np.sum(predicted_values == true_values) / len(true_values)
  print ("Test Accuracy:", test_accuracy)
  #print(classification_report(true_values, predicted_values, target_names=[str(l) for l in label_values]))
  f1 = f1_score(true_values, predicted_values, average='weighted')
  precision = precision_score(true_values, predicted_values, average='weighted')
  recall = recall_score(true_values, predicted_values, average='weighted')

  print(f"F1 Score: {f1:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")


  print(classification_report(true_values, predicted_values))
  print(print(confusion_matrix(true_values, predicted_values)))

  return true_values, predicted_values


import itertools

# plot confusion matrix
# code borrowed from scikit-learn.org
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def count_unique_words(df):
    # Combine all the text in the specified column into a single string
    all_text = ' '.join(df["clean_text"].tolist())

    # Split the text into words
    words = all_text.split()

    # Get the set of unique words
    unique_words = set(words)

    # Return the number of unique words
    return len(unique_words)



if __name__ == "__main__":
        
    
    dataset_name = 'TALLIP' # kaggle, liar_2, liar_6, covid-19
    feature_choice = 'bert' # tfidf, cv, wv, bert
    paraphraser = 'human' #  'parrot' or 'bard', 'gpt','pegasus','llama'
    language = 'English'
    
    
    text_length = 4500
    
    
    
    if dataset_name == 'liar_6':
        num_labels = 6
        
    else:
        num_labels = 2
        
    full_df, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_labels,paraphraser,language)
    
    train_df['clean_text']=train_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
    test_df['clean_text']=test_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
    valid_df['clean_text']=valid_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
        
    total_words = count_unique_words(train_df)

    X = full_df['text'].astype(str)
    y = full_df['label']     
    
    
    
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
    

    
    model, train_dataloader, val_dataloader, test_dataloader, device, tokenizer, test_y = bert_data_prep(X_train, Y_train,X_val,Y_val,X_test,Y_test)
    optimizer = bert_optimizer(model, learning_rate, adam_epsilon)
    scheduler = bert_scheduler(num_epochs, train_dataloader,optimizer)
    model, train_losses, val_losses = train_bert_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device)
    path = "Results/LLMs/"
    #save_bert_model(model, path, tokenizer,train_losses,val_losses)
    model.save_pretrained(f"./Results/LLMs/{dataset_name}_{paraphraser}_{feature_choice}/")
    true_values, predicted_values = test_bert_model(model, test_dataloader, device, test_y)
    cm_test = confusion_matrix(true_values, predicted_values)
    np.set_printoptions(precision=2)
    classification_results['Predicted_Labels'] = predicted_values
    classification_results.to_excel(f"Results/LLMs/{dataset_name}_{paraphraser}_{feature_choice}_classification_results.xlsx")
    
    #%%
    # from transformers_interpret import SequenceClassificationExplainer
    # cls_explainer = SequenceClassificationExplainer(
    #     model,
    #     tokenizer)
    
    # word_attributions = cls_explainer(X_train[0])
    # print(word_attributions)
    # print(cls_explainer.predicted_class_name)
    # cls_explainer.visualize("bert_ner_viz.html")
    # #%%
    # print(X_train[0])
    
