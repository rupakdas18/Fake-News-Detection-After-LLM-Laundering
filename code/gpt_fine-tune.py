import pandas as pd
import load_data
import json
from openai import OpenAI
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


def convert_to_gpt35_format(dataset):
    fine_tuning_data = []
    for _, row in dataset.iterrows():
        json_response = '{"label": "' + row['label'] + '"}'
        fine_tuning_data.append({
            "messages": [
                {"role": "user", "content": row['text']},
                {"role": "assistant", "content": json_response}
            ]
        })
    return fine_tuning_data


def write_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')




def format_test(row):
    formatted_message = [{"role": "user", "content": row['text']}]
    return formatted_message

def predict(test_messages, fine_tuned_model_id):
    response = client.chat.completions.create(
        model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=50
    )
    return response.choices[0].message.content

def store_predictions(test_df, fine_tuned_model_id):
    test_df['Prediction'] = None
    for index, row in test_df.iterrows():
        test_message = format_test(row)
        prediction_result = predict(test_message, fine_tuned_model_id)
        test_df.at[index, 'Prediction'] = prediction_result

    test_df.to_csv("gpt_predictions.csv")




if __name__ == "__main__":
    
    
  dataset_name = 'covid-19' # 'liar', 'kaggle', covid-19
  num_labels = 6 # 2, 6
  paraphraser = 'human' # 'no', 'bard', 'parrot','gpt', 'pegasus', 'llama''
  language = 'English' # 'english', 'german', 'french',
  classifier = 'gpt'
    

    
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

#   train_df = train_df.sample(n=100)
#   test_df = test_df.sample(n=20)
#   valid_df = train_df.sample(n=20)
  
#   text_length = 4500
#   train_df['clean_text']=train_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
#   test_df['clean_text']=test_df['text'].apply(lambda x: load_data.clean_text(x,text_length))
#   valid_df['clean_text']=valid_df['text'].apply(lambda x: load_data.clean_text(x,text_length))

  print(train_df.head())

  converted_data_train = convert_to_gpt35_format(train_df)
  converted_data_test = convert_to_gpt35_format(test_df)
  converted_data_val = convert_to_gpt35_format(valid_df)


  training_file_name = f"{dataset_name}_{paraphraser}_{classifier}train.jsonl"
  #testing_file_name = "test.jsonl"
  validation_file_name = f"{dataset_name}_{paraphraser}_{classifier}valid.jsonl"

  write_to_jsonl(converted_data_train, training_file_name)
  #write_to_jsonl(converted_data_test, testing_file_name)
  write_to_jsonl(converted_data_val, validation_file_name)


  # Upload Training and Validation Files
  training_file = client.files.create(
  file=open(training_file_name, "rb"), purpose="fine-tune")
  validation_file = client.files.create(
  file=open(validation_file_name, "rb"), purpose="fine-tune")
  
  #Fine-tuning
  suffix_name = f"{dataset_name}_{paraphraser}_{classifier}"
  response = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18",
    suffix=suffix_name,)
  print(response)
