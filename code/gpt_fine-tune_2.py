import pandas as pd
import load_data
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW, get_scheduler, GPT2Config
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm.auto import tqdm



    
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label), 'index': idx}

    

# def evaluate(model, data_loader):
#     model.eval()
#     predictions, true_labels = [], []

#     with torch.no_grad():
#         for batch in data_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=-1)
#             predictions.extend(preds.cpu().numpy())
#             true_labels.extend(batch['labels'].cpu().numpy())

#     precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
#     acc = accuracy_score(true_labels, predictions)
#     conf_mat = confusion_matrix(true_labels, predictions)
#     class_report = classification_report(true_labels, predictions)
#     return acc, precision, recall, f1, conf_mat, class_report

def get_predictions(model, data_loader, dataframe):
    model.eval()
    predictions, true_labels, texts = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            texts.extend(dataframe['clean_text'].iloc[batch['index']].tolist())  # Use 'index' to extract text
            batch = {k: v.to(device) for k, v in batch.items() if k != 'index'}  # Pass only relevant data to the model
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    acc = accuracy_score(true_labels, predictions)
    conf_mat = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions)

    print(f'Accuracy: {acc}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')
    print('Confusion Matrix:\n', conf_mat)
    print('Classification Report:\n', class_report)

    return texts, predictions, true_labels




if __name__ == "__main__":




    dataset_name = 'covid-19' # 'liar', 'kaggle', covid-19
    num_labels = 2 # 2, 6
    paraphraser = 'human' # 'gpt', 'pegasus', 'llama''
    language = 'English' # 'english', 'german', 'french',
    classifier = 'gpt_2'



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load configuration and modify it to acknowledge the new padding token
    config = GPT2Config.from_pretrained('gpt2', pad_token_id=tokenizer.pad_token_id, num_labels=num_labels)

    # Load the model with updated configuration
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config)

    # Update model's tokenizer to include the padding token
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)


    
    
        

        
    full_df, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_labels,paraphraser,language,classifier)

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

    text_length = 512
    train_df['clean_text']=train_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
    test_df['clean_text']=test_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
    valid_df['clean_text']=valid_df['text'].apply(lambda x: load_data.clean_text(x,text_length))

    # train_df = train_df.sample(n=100)
    # test_df = test_df.sample(n=20)
    # valid_df = train_df.sample(n=20)

    print(train_df.head())

    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_df, tokenizer, max_length=512)
    valid_dataset = TextDataset(valid_df, tokenizer, max_length=512)
    test_dataset = TextDataset(test_df, tokenizer, max_length=512)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    print("Data loaders created: ", len(train_loader))


    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


   # Training loop
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc="Epoch {:1d}".format(epoch+1))
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'index'}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': loss.item()})


    # Evaluate the model
    # accuracy, precision, recall, f1_score, conf_matrix, class_rep = evaluate(model, test_loader)
    # print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}')
    # print('Confusion Matrix:\n', conf_matrix)
    # print('Classification Report:\n', class_rep)

    texts, predicted_labels, actual_labels = get_predictions(model, test_loader, test_df)
    results_df = pd.DataFrame({
        'Text': texts,
        'Predicted Label': predicted_labels,
        'Actual Label': actual_labels
    })

    # Save to CSV
    results_df.to_csv(f"gpt2_pred_{dataset_name}_{paraphraser}.csv", index=False)