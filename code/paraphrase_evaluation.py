# -*- coding: utf-8 -*-

'''
BERT_score
'''

import pandas as pd
import bert_score
import matplotlib as plt


dataset = 'covid_19'
paraphraser = 'GPT'

df = pd.read_csv(f"{dataset}/{dataset}_{paraphraser}_paraphrased_merged_2.csv")

print(df)
df=df.dropna()

original_sentences = df['text'].tolist()
#original_sentences = original_sentences[:500]
paraphrased_sentences = df['GPT'].tolist() # Select the model
#paraphrased_sentences = paraphrased_sentences[:500]

# Compute BERTScore for each pair of original and paraphrased sentences
P, R, F1 = bert_score.score(original_sentences, paraphrased_sentences, lang="en", model_type='roberta-large-mnli', rescale_with_baseline=True, verbose=True)




P_list = [round(num, 4) for num in P.tolist()]
R_list = [round(num, 4) for num in R.tolist()]
F1_list = [round(num, 4) for num in F1.tolist()]

for i, val in enumerate(F1_list):
  df.loc[i, "bertscore_p"] = P_list[i]
  df.loc[i, "bertscore_r"] = R_list[i]
  df.loc[i, "bertscore_f1"] = F1_list[i]


# Extract the desired columns into a new dataframe
#df_selected = df[['id', 'text', 't5_0', 'bertscore_p', 'bertscore_r', 'bertscore_f1', 'bertscore_p_rescale', 'bertscore_r_rescale', 'bertscore_f1_rescale']]
df_selected = df[['id', 'text', 'GPT', 'bertscore_p', 'bertscore_r', 'bertscore_f1']]


# Export the selected columns to a CSV file
df_selected.to_csv(f'{dataset}/Evaluation/{dataset}_{paraphraser}_bertscore_merged_2.csv', index=False)

#%%

'''
ROUGE
'''

import evaluate
import pandas as pd

dataset = 'liar'
paraphraser = 'GPT'

df = pd.read_csv(f"{dataset}/{dataset}_{paraphraser}_paraphrased_valid.csv") 

rouge = evaluate.load("rouge")

for i, ref in df["text"].items():
  references = [ref]
  predictions = [df.at[i, 'GPT']]

  score = rouge.compute(predictions=predictions, references=references)

  df.loc[i, 'rouge1'] = round(score['rouge1'], 4)
  df.loc[i, 'rouge2'] = round(score['rouge2'], 4)
  df.loc[i, 'rougeL'] = round(score['rougeL'], 4)
     

# Extract the desired columns into a new dataframe
df_selected = df[['id', 'text', 'GPT', 'rouge1-p', 'rouge1-r', 'rouge1-f', 'rouge2-p', 'rouge2-r', 'rouge2-f', 'rougeL-p', 'rougeL-r', 'rougeL-f']]

# Export the selected columns to a CSV file
df_selected.to_csv('f{dataset}/Evaluation/{dataset}_{paraphraser}_rouge_valid.csv', index=False)



#%%

'''
Merge files
'''



import pandas as pd
import os

paraphraser = 'GPT'

# Load the CSV files into dataframes
df1 = pd.read_csv(f"covid_19/covid-19_{paraphraser}_paraphrased_train.csv")
df2 = pd.read_csv(f"covid_19/covid-19_{paraphraser}_paraphrased_test.csv")
df3 = pd.read_csv(f"covid_19/covid-19_{paraphraser}_paraphrased_valid.csv")

# Merge the dataframes by concatenating them
merged_df = pd.concat([df1, df2, df3], ignore_index=True)


# Save the merged dataframe into a single CSV file (optional, if you want to store it)
merged_df.to_csv(f"covid_19/covid-19_{paraphraser}_paraphrased_merged_2.csv", index=False)


#%%

'''
Check the distribution of the BERTScore (1 row 3 column)
'''


import pandas as pd
import matplotlib.pyplot as plt

# Paths to the 4 CSV files (replace these with actual paths)

dataset = 'liar' # covid_19, liar


# file_paths = [
#     f'{dataset}/Evaluation/{dataset}_GPT_bertscore_merged.csv',
#     f'{dataset}/Evaluation/{dataset}_llama_bertscore_merged.csv',
#     f'{dataset}/Evaluation/{dataset}_pegasus_bertscore_merged.csv',
#     f'{dataset}/Evaluation/{dataset}_parrot_bertscore_merged.csv'
# ]

paraphrasers = ['GPT','Llama','Pegasus']

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 4))

# Plot each file in a separate subplot
for i, paraphraser in enumerate(paraphrasers):
    file_path = f'{dataset}/Evaluation/{dataset}_{paraphraser}_bertscore_merged.csv'

    df = pd.read_csv(file_path)
    
    # Select the appropriate subplot
    #ax = axes[i // 2, i % 2]
    ax = axes[i]
    
    # Plot the histogram for "bert_f1"
    ax.hist(df['bertscore_f1'], bins=30, color='blue', edgecolor='black', alpha=0.7)
    
    #ax.set_title(f'Distribution of bertscore F1 - File {i + 1}')
    ax.set_xlabel(f'{paraphraser}',fontsize=12)
    ax.set_ylabel('Frequency',fontsize=12)
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# Adjust layout
plt.tight_layout()

fig.savefig(f'{dataset}_bert_f1_distribution.pdf', format='pdf', bbox_inches='tight')

#%%

'''
Check the distribution of the BERTScore (3 row 1 column)
'''



import pandas as pd
import matplotlib.pyplot as plt

# Dataset name and paraphrasers
dataset = 'liar'  # Choose between 'covid_19', 'liar'
paraphrasers = ['GPT', 'Llama', 'Pegasus']
colors = ["#005280", "#CE7EAA", "#BCE4C0"]

# Initialize global min and max values for x-axis limits
global_min = float('inf')
global_max = float('-inf')

# Step 1: First pass - Find the global min and max across all files
for paraphraser in paraphrasers:
    file_path = f'{dataset}/Evaluation/{dataset}_{paraphraser}_bertscore_merged.csv'
    df = pd.read_csv(file_path)
    
    # Update global min and max
    global_min = min(global_min, df['bertscore_f1'].min())
    global_max = max(global_max, df['bertscore_f1'].max())

# Step 2: Create subplots and plot histograms with the same x-axis scale
fig, axes = plt.subplots(3, 1, figsize=(6, 3))  # Adjust height for better spacing

for i, paraphraser in enumerate(paraphrasers):
    file_path = f'{dataset}/Evaluation/{dataset}_{paraphraser}_bertscore_merged.csv'
    df = pd.read_csv(file_path)
    
    ax = axes[i]
    ax.hist(df['bertscore_f1'], bins=30, color=colors[i], edgecolor='black', alpha=0.7)
    
    # Set consistent x-axis limits across all subplots
    ax.set_xlim(global_min, global_max)
    
    # Labeling
    ax.set_xlabel(f'{paraphraser}', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

fig.savefig(f'{dataset}_bert_f1_distribution.pdf', format='pdf', bbox_inches='tight')

#%%

'''
Result comparison

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from the Excel file (update the file path accordingly)
file_path = 'All_Results.xlsx'
sheetname='Covid-19'
#sheetname='Liar-6', 'Covid-19'
df = pd.read_excel(file_path,sheet_name=sheetname)
colors = ["#000000", "#005280", "#CE7EAA", "#BCE4C0"]

print(df)

pipelines_order  = [
'BERT',
'T5',
'Llama',
'GPT-2',
'CNN',
'LSTM',
'SVM-cv',
'SVM-tfidf',
'SVM-wv',
'LR-cv',
'LR-tfidf',
'LR-wv',
'RF-cv',
'RF-tfidf',
'RF-wv',
'DT-cv',
'DT-tfidf',
'DT-wv'
 ]

paraphraser_order = ['Human','GPT','Llama','Pegasus']


df = df[df['Paraphraser'].isin(paraphraser_order)]
df = df[df['Pipeline'].isin(pipelines_order)]


# Filter the data based on the pipelines in the desired order
df['Pipeline'] = pd.Categorical(df['Pipeline'], categories=pipelines_order, ordered=True)
df['Paraphraser'] = pd.Categorical(df['Paraphraser'], categories=paraphraser_order, ordered=True)
df = df.sort_values(['Pipeline', 'Paraphraser'])

# Grouping the data by 'Pipeline' and getting the 'F1' values for each pipeline
pipelines = df['Pipeline'].unique()
paraphraser = df['Paraphraser'].unique()




print(paraphraser)
# Initialize a dictionary to hold F1 scores for each pipeline
# f1_scores = {pipeline: [] for pipeline in pipelines}

f1_scores = {dataset: [] for dataset in paraphraser}
for dataset in paraphraser:
    f1_scores[dataset] = df[df['Paraphraser'] == dataset]['F1'].values




# Populate the F1 scores for each pipeline
# for pipeline in pipelines:
#     f1_scores[pipeline] = df[df['Pipeline'] == pipeline]['F1'].values

print(f1_scores)




# Set up the figure and axes for the plot
fig, ax = plt.subplots(figsize=(15, 6))

# Set the number of pipelines and bar width
n_pipelines = len(pipelines)
bar_width = 0.2
index = np.arange(n_pipelines)

# Loop through datasets to plot each group of bars
for i, dataset in enumerate(paraphraser):
    ax.bar(index + i * bar_width, f1_scores[dataset], bar_width, label=dataset,color = colors[i])

# Labeling
ax.set_xlabel('Pipeline', fontsize = 20)
ax.set_ylabel('F1 Score',fontsize = 20)
#ax.set_title('F1 Scores by Dataset Grouped by Pipelines')
ax.set_xticks(index + bar_width * (len(paraphraser) - 1) / 2)
ax.set_xticklabels(pipelines, rotation=45)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Add legend
ax.legend(loc='lower right', fontsize = 20)

# Display the plot
plt.tight_layout()
plt.show()

fig.savefig(f'{sheetname}_comparison.pdf', format='pdf', bbox_inches='tight')


#%%

"""
Shapiro-Wilk Test (Statistical Test) to find out the normality
"""

import pandas as pd
from scipy.stats import shapiro

# Dataset name and paraphrasers
dataset = 'liar'  # Choose between 'covid_19', 'liar'
paraphrasers = ['GPT', 'Llama', 'Pegasus']



GPT_bertscore = pd.read_csv(f'{dataset}/Evaluation/{dataset}_GPT_bertscore_merged.csv') 
llama_bertscore = pd.read_csv(f'{dataset}/Evaluation/{dataset}_llama_bertscore_merged.csv') 
pegasus_bertscore = pd.read_csv(f'{dataset}/Evaluation/{dataset}_pegasus_bertscore_merged.csv') 

GPT_bertscore = GPT_bertscore.dropna()
llama_bertscore = llama_bertscore.dropna()
pegasus_bertscore = pegasus_bertscore.dropna()

print(llama_bertscore['bertscore_f1'])


# Perform Shapiro-Wilk test for each paraphraser
stat_gpt, p_gpt = shapiro(GPT_bertscore['bertscore_f1'])
stat_llama, p_llama = shapiro(llama_bertscore['bertscore_f1'])
stat_pegasus, p_pegasus = shapiro(pegasus_bertscore['bertscore_f1'])


print(f"GPT: W={stat_gpt}, p={p_gpt}")
print(f"Llama: W={stat_llama}, p={p_llama}")
print(f"Pegasus: W={stat_pegasus}, p={p_pegasus}")

# Interpretation
if p_gpt > 0.05:
    print("GPT distribution normal")
else:
    print("not normal.")
    
if p_llama > 0.05:
    print("Llama distribution normal")
else:
    print("not normal.")
    
if p_pegasus > 0.05:
    print("Pegasus distribution normal")
else:
    print("not normal.")
    
#%%

import scipy.stats as stats

stats.levene(GPT_bertscore['bertscore_f1'],llama_bertscore['bertscore_f1'],pegasus_bertscore['bertscore_f1'], center='mean')

#%%


from scipy.stats import kruskal

# Assuming the data is loaded into DataFrames: df_gpt, df_llama, df_pegasus

# Perform Kruskal-Wallis H test
# h_stat, p_value = kruskal(GPT_bertscore['bertscore_f1'], 
#                           llama_bertscore['bertscore_f1'], 
#                           pegasus_bertscore['bertscore_f1'])




h_stat_1, p_value_1 = kruskal(GPT_bertscore['bertscore_f1'], 
                          llama_bertscore['bertscore_f1'])

h_stat_2, p_value_2 = kruskal(GPT_bertscore['bertscore_f1'], 
                          pegasus_bertscore['bertscore_f1'])

h_stat_3, p_value_3 = kruskal(llama_bertscore['bertscore_f1'], 
                          pegasus_bertscore['bertscore_f1'])

#print(f"H-statistic: {h_stat}, p-value: {p_value}")

print(p_value_1,p_value_2,p_value_3)

# Interpretation
if p_value_1 < 0.05:
    print("There is a statistically significant difference between GPT and llama.")
else:
    print("There is no a statistically significant difference between GPT and llama.")

if p_value_2 < 0.05:
    print("There is a statistically significant difference between GPT and Pegasus.")
else:
    print("There is no a statistically significant difference between GPT and Pegasus.")
    
if p_value_3 < 0.05:
    print("There is a statistically significant difference between pegasus and llama.")
else:
    print("There is no a statistically significant difference between pegasus and llama.")    






#%%

"""
Welch’s t-Test in Python

"""


import scipy.stats as stats 
import numpy as np 
  
# Creating data groups 
data_group1 = np.array(GPT_bertscore['bertscore_f1']) 

data_group2 = np.array(llama_bertscore['bertscore_f1']) 

data_group1 = np.array(GPT_bertscore['bertscore_f1']) 

data_group3 = np.array(pegasus_bertscore['bertscore_f1']) 
  
# Conduct Welch's t-Test and print the result 
print(stats.ttest_ind(data_group2, data_group3, equal_var = False)) 

#%%


import numpy as np
import pandas as pd
import researchpy as rp
from math import sqrt

dataset = 'liar'

GPT_bertscore = pd.read_csv(f'{dataset}/Evaluation/{dataset}_GPT_bertscore_merged.csv') 
llama_bertscore = pd.read_csv(f'{dataset}/Evaluation/{dataset}_llama_bertscore_merged.csv') 
pegasus_bertscore = pd.read_csv(f'{dataset}/Evaluation/{dataset}_pegasus_bertscore_merged.csv') 

GPT_bertscore = GPT_bertscore.dropna()
llama_bertscore = llama_bertscore.dropna()
pegasus_bertscore = pegasus_bertscore.dropna()


def hedges_g(group1, group2):
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Calculate pooled standard deviation
    pooled_std = sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    cohen_d = (mean1 - mean2) / pooled_std
    
    # Apply correction factor for small sample sizes
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g_value = cohen_d * correction
    
    return hedges_g_value

# Example usage: separate data into two groups
group1 = GPT_bertscore['bertscore_f1']
group2 = llama_bertscore['bertscore_f1']
group3 = pegasus_bertscore['bertscore_f1']

# Calculate Hedge's g for the two groups
g_value1 = hedges_g(group1, group2)
g_value2 = hedges_g(group1, group3)
g_value3 = hedges_g(group2, group3)

print(f"Hedge's g of GPT and Llama: {g_value1}")
print(f"Hedge's g GPT and Pegasus: {g_value2}")
print(f"Hedge's g Llama and Pegasus: {g_value3}")


#%%

from weasyprint import HTML

# Set path to wkhtmltopdf binary (only required on Windows)
path_to_wkhtmltopdf = r'Figure/liar_6_gpt_LSTM_sample_1240.html'
HTML('Figure/liar_6_gpt_LSTM_sample_1240.html').write_pdf('output.pdf')







