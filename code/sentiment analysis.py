# -*- coding: utf-8 -*-




#%%

import pandas as pd
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings("ignore")


# For Windows
# Check if CUDA is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # Use CUDA
#     print("Using CUDA on device:", torch.cuda.get_device_name(torch.cuda.current_device()))
# else:
#     device = torch.device("cpu")  # Use CPU if CUDA is not available
#     print("CUDA is not available, using CPU instead.")


# For MAC
device = torch.device("mps")
print(f"working on {device}")



def find_sentiment(text):
    
    try:
        distilled_student_sentiment_classifier = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
            return_all_scores=True, device = device)
        
        result = distilled_student_sentiment_classifier(text)
        
        return float(result[0][0]['score']), float(result[0][1]['score']), float(result[0][2]['score'])
    
    except Exception:
        return 0, 0, 0  # Return zero for all sentiments in case of an error
    
         
print(find_sentiment("The conversation with my mom was heartbreaking"))
print(find_sentiment("The conversation with my dad was heartbreaking"))
#%%


dataset = 'liar' # covid-19
paraphraser = 'GPT'

dataframe = pd.read_csv(f"{dataset}/Evaluation/{dataset}_{paraphraser}_bertscore_merged.csv")
human_text = dataframe['text']
paraphrased_text = dataframe[f"{paraphraser}"]

print(human_text)
print(paraphrased_text)


dataframe[["h_sen_pos", "h_sen_neu", "h_sen_neg"]] = dataframe['text'].apply(lambda x: find_sentiment(x)).apply(pd.Series)
dataframe[["p_sen_pos", "p_sen_neu", "p_sen_neg"]] = dataframe[f"{paraphraser}"].apply(lambda x: find_sentiment(x)).apply(pd.Series)

dataframe.to_csv(f"Sentiment_analysis/{dataset}_{paraphraser}_sentiment_2.csv")

#%%


import pandas as pd
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA
    print("Using CUDA on device:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = torch.device("cpu")  # Use CPU if CUDA is not available
    print("CUDA is not available, using CPU instead.")
    
    

def find_sentiment(text):
    try:
        distilled_student_sentiment_classifier = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            return_all_scores=True, device = device
        )
        result = distilled_student_sentiment_classifier(text)
        return float(result[0][0]['score']), float(result[0][1]['score']), float(result[0][2]['score'])
    except Exception:
        return 0, 0, 0  # Return zero for all sentiments in case of an error

# Load your data
dataset = 'liar' # Adjust as needed for 'covid-19' or other datasets
paraphraser = 'GPT'
dataframe = pd.read_csv(f"{dataset}/Evaluation/{dataset}_{paraphraser}_bertscore_merged.csv")

# Initialize columns for storing sentiment scores
dataframe[["h_sen_pos", "h_sen_neu", "h_sen_neg"]] = pd.DataFrame([[0.0, 0.0, 0.0]], index=dataframe.index)
dataframe[["p_sen_pos", "p_sen_neu", "p_sen_neg"]] = pd.DataFrame([[0.0, 0.0, 0.0]], index=dataframe.index)

# Apply sentiment analysis row by row
for index, row in dataframe.iterrows():
    h_scores = find_sentiment(row['text'])
    p_scores = find_sentiment(row[paraphraser])
    dataframe.loc[index, ["h_sen_pos", "h_sen_neu", "h_sen_neg"]] = h_scores
    dataframe.loc[index, ["p_sen_pos", "p_sen_neu", "p_sen_neg"]] = p_scores

# Save the updated dataframe
dataframe.to_csv(f"Sentiment_analysis/{dataset}_{paraphraser}_sentiment.csv")


#%%

'''
Sentiment shift analysis
'''
import pandas as pd
import matplotlib.pyplot as plt

dataset = 'liar'
paraphraser = 'GPT'

GPT_dataframe = pd.read_csv(f'Sentiment_analysis/{dataset}_{paraphraser}_sentiment.csv')

print(GPT_dataframe.columns)

bertscore_f1 = GPT_dataframe['bertscore_f1']
sentiment_shift = GPT_dataframe['sentiment_shift']

print(sentiment_shift)


# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(bertscore_f1, sentiment_shift, color='blue', marker='+')

# Add labels and title
plt.xlabel('BERTScore F1', fontsize=12)
plt.ylabel('Sentiment Shift', fontsize=12)
#plt.title('Scatter Plot of BERTScore F1 vs Sentiment Shift', fontsize=14)

# Show plot
plt.grid(True)
plt.show()

plt.savefig(f'Figures/{dataset}_{paraphraser}_sentiment_shift.pdf', format='pdf', bbox_inches='tight')

#%%

import pandas as pd
import matplotlib.pyplot as plt

# Load your dataframe
dataset = 'covid_19'
paraphraser = 'GPT'
GPT_dataframe = pd.read_csv(f'Sentiment_analysis/{dataset}_{paraphraser}_sentiment.csv')

# Print column names and specific column data
print(GPT_dataframe.columns)
print(GPT_dataframe['sentiment_shift'])

# Extracting columns for the scatter plot
bertscore_f1 = GPT_dataframe['bertscore_f1']
sentiment_shift = GPT_dataframe['sentiment_shift']

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(6, 4))

# Create scatter plot
ax.scatter(bertscore_f1, sentiment_shift, color='blue', marker='+', alpha=0.5)

# Add labels
ax.set_xlabel('$F1_{BERT}$-score', fontsize=12)
ax.set_ylabel('Sentiment Shift', fontsize=12)
# ax.set_title('Scatter Plot of BERTScore F1 vs Sentiment Shift', fontsize=14)

# Show grid
ax.grid(True)

# Save the plot as a PDF file
fig.savefig(f'{dataset}_{paraphraser}_sentiment_shift.pdf', format='pdf', bbox_inches='tight')

# Show plot
plt.show()


#%%

import pandas as pd
import matplotlib.pyplot as plt

dataset = 'liar'
paraphraser = 'GPT'

GPT_dataframe = pd.read_excel(f'Sentiment_analysis/liar_GPT_LSTM_sentiment_test.xlsx')

print(GPT_dataframe)

bertscore_f1_true = GPT_dataframe[GPT_dataframe['prediction'] == True]['bertscore_f1'].tolist()
sentiment__true = GPT_dataframe[GPT_dataframe['prediction'] == True]['bertscore_f1'].tolist()


print(len(bertscore_f1_true))
#print(len(bertscore_f1_false))

