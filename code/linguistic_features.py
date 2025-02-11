# Lexical Diversity

import pandas as pd
import load_data
import nltk
import textstat
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import spacy

def dataset_statistics(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus):

    datasets = {
    "Human": full_df_human,
    "GPT": full_df_gpt,
    "LLaMA": full_df_llama,
    "Pegasus": full_df_pegasus
}
    
    # Print statistics for each dataset
    for label, data in datasets.items():
        mean_value = np.mean(data)
        median_value = np.median(data)
        range_value = np.ptp(data)  # Peak-to-peak (max-min)
        std_dev = np.std(data)
        
        print(f"Statistics for {label}:")
        print(f"  Mean: {mean_value:.4f}")
        print(f"  Median: {median_value:.4f}")
        print(f"  Range: {range_value:.4f}")
        print(f"  Standard Deviation: {std_dev:.4f}")
        print("-" * 40)


# Function to filter out extreme negative scores
def filter_readability(df,threshold=-100):
    return df[df['readability'] > threshold]['readability']

def filter_grade_level(df,threshold=40):
    return df[df['Grade_Level'] < threshold]['Grade_Level']

def filter_depndency_depth(df,threshold=40):
    return df[df['dependency_tree_height'] < threshold]['dependency_tree_height']


def plot_readability_distribution(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column):

    # Calculate readability scores for each version
    full_df_human['readability'] = full_df_human[column].apply(textstat.flesch_reading_ease)
    full_df_gpt['readability'] = full_df_gpt[column].apply(textstat.flesch_reading_ease)
    full_df_llama['readability'] = full_df_llama[column].apply(textstat.flesch_reading_ease)
    full_df_pegasus['readability'] = full_df_pegasus[column].apply(textstat.flesch_reading_ease)


    # Apply filtering
    readability_human = filter_readability(full_df_human)
    readability_gpt = filter_readability(full_df_gpt)
    readability_llama = filter_readability(full_df_llama)
    readability_pegasus = filter_readability(full_df_pegasus)



    # Create subplots for each readability distribution
    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True, sharey=True)

    # Define bin edges for consistency
    bins = np.linspace(
        min(readability_human.min(), readability_gpt.min(), readability_llama.min(), readability_pegasus.min()),
        max(readability_human.max(), readability_gpt.max(), readability_llama.max(), readability_pegasus.max()), 
        30
    )

    # Plot each readability distribution separately but with the same scale
    axes[0].hist(readability_human, bins=bins, alpha=0.7, edgecolor='black', color='#000000')
    axes[0].set_title('Human')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(readability_gpt, bins=bins, alpha=0.7, edgecolor='black', color='#005280')
    axes[1].set_title('GPT')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(readability_llama, bins=bins, alpha=0.7, edgecolor='black', color='#CE7EAA')
    axes[2].set_title('LLaMA')
    axes[2].set_ylabel('Frequency')

    axes[3].hist(readability_pegasus, bins=bins, alpha=0.7, edgecolor='black', color='#BCE4C0')
    axes[3].set_title('Pegasus')
    axes[3].set_ylabel('Frequency')
    axes[3].set_xlabel('Flesch Reading Ease Score')

    # Set the same x and y limits for all plots to make them comparable
    for ax in axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()


    pdf_filename = f"{dataset_name}_readability_comparison.pdf"
    plt.savefig(pdf_filename, format="pdf")

    plt.show()

    dataset_statistics(readability_human,readability_gpt,readability_llama,readability_pegasus)

    return full_df_human, full_df_gpt, full_df_llama, full_df_pegasus


def plot_grade_level_distribution(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column):

    # Calculate The Flesch-Kincaid Grade Level for each version
    full_df_human['Grade_Level'] = full_df_human[column].apply(textstat.flesch_kincaid_grade)
    full_df_gpt['Grade_Level'] = full_df_gpt[column].apply(textstat.flesch_kincaid_grade)
    full_df_llama['Grade_Level'] = full_df_llama[column].apply(textstat.flesch_kincaid_grade)
    full_df_pegasus['Grade_Level'] = full_df_pegasus[column].apply(textstat.flesch_kincaid_grade)

    

    # Extract Grade level scores
    grade_level_human = filter_grade_level(full_df_human)
    grade_level_gpt = filter_grade_level(full_df_gpt)
    grade_level_llama = filter_grade_level(full_df_llama)
    grade_level_pegasus = filter_grade_level(full_df_pegasus)

    

# Create subplots for each Grade label distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    # Define bin edges for consistency
    bins = np.linspace(
        min(grade_level_human.min(), grade_level_gpt.min(), grade_level_llama.min(), grade_level_pegasus.min()),
        max(grade_level_human.max(), grade_level_gpt.max(), grade_level_llama.max(), grade_level_pegasus.max()), 
        30
    )

    # Plot each Grade level separately but with the same scale
    axes[0, 0].hist(grade_level_human, bins=bins, alpha=0.7, edgecolor='black', color='blue')
    axes[0, 0].set_title('Human Grade level')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(grade_level_gpt, bins=bins, alpha=0.7, edgecolor='black', color='red')
    axes[0, 1].set_title('GPT Grade level')

    axes[1, 0].hist(grade_level_llama, bins=bins, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].set_title('LLaMA Grade level')
    axes[1, 0].set_xlabel('Flesch Grade level')
    axes[1, 0].set_ylabel('Frequency')

    axes[1, 1].hist(grade_level_pegasus, bins=bins, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].set_title('Pegasus Grade level')
    axes[1, 1].set_xlabel('Flesch Grade level')

    # Set the same x and y limits for all plots to make them comparable
    for ax in axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    

    pdf_filename = f"{dataset_name}_Grade level_comparison.pdf"
    plt.savefig(pdf_filename, format="pdf")

    plt.show()

    return full_df_human, full_df_gpt, full_df_llama, full_df_pegasus



def plot_readability_distribution_2():

    threshold = -200
    # Apply filtering
    readability_human = filter_readability(full_df_human)
    readability_gpt = filter_readability(full_df_gpt)
    readability_llama = filter_readability(full_df_llama)
    readability_pegasus = filter_readability(full_df_pegasus)



    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(readability_human, bins=30, alpha=0.6, label='Human', edgecolor='black')
    plt.hist(readability_gpt, bins=30, alpha=0.6, label='GPT', edgecolor='black')
    plt.hist(readability_llama, bins=30, alpha=0.6, label='LLaMA', edgecolor='black')
    plt.hist(readability_pegasus, bins=30, alpha=0.6, label='Pegasus', edgecolor='black')

    # Labels and legend
    plt.xlabel('Flesch Reading Ease Score')
    plt.ylabel('Frequency')
    plt.title('Readability Score Distribution for Different Text Versions')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()


def complex_words_ratio(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column):

    def complex_word_ratio(text):
        if isinstance(text, str) and len(text.split()) > 0:
            return textstat.difficult_words(text) / len(text.split())
        else:
            return 0  # Handle empty or invalid text

    # Calculate complex word ratio for each version
    full_df_human['complex_word_ratio'] = full_df_human[column].apply(complex_word_ratio)
    full_df_gpt['complex_word_ratio'] = full_df_gpt[column].apply(complex_word_ratio)
    full_df_llama['complex_word_ratio'] = full_df_llama[column].apply(complex_word_ratio)
    full_df_pegasus['complex_word_ratio'] = full_df_pegasus[column].apply(complex_word_ratio)

    # Extract complex word ratios
    complex_word_ratio_human = full_df_human['complex_word_ratio']
    complex_word_ratio_gpt = full_df_gpt['complex_word_ratio']
    complex_word_ratio_llama = full_df_llama['complex_word_ratio']
    complex_word_ratio_pegasus = full_df_pegasus['complex_word_ratio']

    # Create subplots for each Grade label distribution
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True)

    # Define bin edges for consistency
    bins = np.linspace(
        min(complex_word_ratio_human.min(), complex_word_ratio_gpt.min(), complex_word_ratio_llama.min(), complex_word_ratio_pegasus.min()),
        max(complex_word_ratio_human.max(), complex_word_ratio_gpt.max(), complex_word_ratio_llama.max(), complex_word_ratio_pegasus.max()), 
        30
    )

    # Plot each Grade level separately but with the same scale
    axes[0, 0].hist(complex_word_ratio_human, bins=bins, alpha=0.7, edgecolor='black', color='blue')
    axes[0, 0].set_title('Human complexity ration')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(complex_word_ratio_gpt, bins=bins, alpha=0.7, edgecolor='black', color='red')
    axes[0, 1].set_title('GPT complexity ration')

    axes[0, 2].hist(complex_word_ratio_llama, bins=bins, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].set_title('LLaMA complexity ration')
    axes[0, 2].set_xlabel('complexity ration')
    axes[0, 2].set_ylabel('Frequency')

    axes[0, 3].hist(complex_word_ratio_pegasus, bins=bins, alpha=0.7, edgecolor='black', color='purple')
    axes[0, 3].set_title('Pegasus complexity ration')
    axes[0, 3].set_xlabel('Flesch complexity ration')

    # Set the same x and y limits for all plots to make them comparable
    for ax in axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    

    pdf_filename = f"{dataset_name}_complex_word_ratio.pdf"
    plt.savefig(pdf_filename, format="pdf")

    plt.show()

    return full_df_human, full_df_gpt, full_df_llama, full_df_pegasus


import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# def compute_dependency_tree_height(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column):
#     """
#     Computes the maximum height of the dependency tree for each row in a DataFrame.

#     :param df: Pandas DataFrame containing text data.
#     :param text_column: Column name containing text.
#     :return: DataFrame with an additional 'dependency_tree_height' column.
#     """

#     # Function to compute height of a token
#     def get_height(token):
#         if not list(token.children):  # If leaf node, height = 0
#             return 0
#         return 1 + max(get_height(child) for child in token.children)

#     # Function to compute tree height for a given sentence
#     def get_tree_height(sentence):
#         doc = nlp(sentence)
#         root = [token for token in doc if token.head == token][0]  # Find root token
#         return get_height(root)  # Compute max height from the root

def compute_dependency_tree_depth(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column):

    def get_tree_height(text):
        if isinstance(text, str) and len(text.split()) > 0:
            doc = nlp(text)
            return max([max(get_depth(token) for token in sent.root.subtree) for sent in doc.sents])
        else:
            return 0

    def get_depth(token):
        if not list(token.children):
            return 1
        return 1 + max(get_depth(child) for child in token.children)


    # Calculate dependency tree depth for each version
    full_df_human['dependency_tree_height'] = full_df_human[column].apply(get_tree_height)
    full_df_gpt['dependency_tree_height'] = full_df_gpt[column].apply(get_tree_height)
    full_df_llama['dependency_tree_height'] = full_df_llama[column].apply(get_tree_height)
    full_df_pegasus['dependency_tree_height'] = full_df_pegasus[column].apply(get_tree_height)


    # Apply filtering
    dependency_depth_human = filter_depndency_depth(full_df_human)
    dependency_depth_gpt = filter_depndency_depth(full_df_gpt)
    dependency_depth_llama = filter_depndency_depth(full_df_llama)
    dependency_depth_pegasus = filter_depndency_depth(full_df_pegasus)


    # Create subplots for each Grade label distribution
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True, sharey=True)
    # Define bin edges for consistency
    bins = np.linspace(
        min(dependency_depth_human.min(), dependency_depth_gpt.min(), dependency_depth_llama.min(), dependency_depth_pegasus.min()),
        max(dependency_depth_human.max(), dependency_depth_gpt.max(), dependency_depth_llama.max(), dependency_depth_pegasus.max()), 
        10
    )

    # Plot each dependency tree depth separately but with the same scale
    axes[0].hist(dependency_depth_human, bins=bins, alpha=0.7, edgecolor='black', color='#000000')
    axes[0].set_title('Human')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(dependency_depth_gpt, bins=bins, alpha=0.7, edgecolor='black', color='#005280')
    axes[1].set_title('GPT')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(dependency_depth_llama, bins=bins, alpha=0.7, edgecolor='black', color='#CE7EAA')
    axes[2].set_title('LLaMA')
    axes[2].set_ylabel('Frequency')

    axes[3].hist(dependency_depth_pegasus, bins=bins, alpha=0.7, edgecolor='black', color='#BCE4C0')
    axes[3].set_title('Pegasus')
    axes[3].set_ylabel('Frequency')
    axes[3].set_xlabel('Depth of parse tree')

    # Set the same x and y limits for all plots to make them comparable
    for ax in axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    

    pdf_filename = f"{dataset_name}_dependency_depth_distribution.pdf"
    plt.savefig(pdf_filename, format="pdf")

    plt.show()

    dataset_statistics(dependency_depth_human,dependency_depth_gpt,dependency_depth_llama,dependency_depth_pegasus)

    return full_df_human, full_df_gpt, full_df_llama, full_df_pegasus





   
if __name__=="__main__":

    dataset_name = 'covid-19' # 'TALLIP''liar_2','liar_6', 'kaggle', 'covid-19'
    project = "02-Paraphrasing"
    num_class = 2

    full_df_human, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_class,"human","English","doesn't matter")
    full_df_gpt, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_class,"gpt","English","doesn't matter")
    full_df_llama, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_class,"llama","English","doesn't matter")
    full_df_pegasus, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_class,"pegasus","English","doesn't matter")


    full_df_human = full_df_human.dropna()
    full_df_gpt = full_df_gpt.dropna()
    full_df_llama = full_df_llama.dropna()
    full_df_pegasus = full_df_pegasus.dropna()
    
    text_length = 1200
    # full_df_human['clean_text']=full_df_human['text'].apply(lambda x: load_data.clean_text(x,text_length))
    # full_df_gpt['clean_text']=full_df_gpt['text'].apply(lambda x: load_data.clean_text(x,text_length))
    # full_df_llama['clean_text']=full_df_llama['text'].apply(lambda x: load_data.clean_text(x,text_length))
    # full_df_pegasus['clean_text']=full_df_pegasus['text'].apply(lambda x: load_data.clean_text(x,text_length))
    # column = 'clean_text'
    column = 'text'


    print("size of the human-written dataset: ", full_df_human.shape)
    print("size of the gpt-written dataset: ", full_df_gpt.shape)
    print("size of the llama-written dataset: ", full_df_llama.shape)
    print("size of the pegasus-written dataset: ", full_df_pegasus.shape)


    print(full_df_human.columns)
    print(full_df_gpt.columns)
    print(full_df_llama.columns)
    print(full_df_pegasus.columns)


    
    full_df_human, full_df_gpt, full_df_llama, full_df_pegasus = plot_readability_distribution(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column)
    # full_df_human, full_df_gpt, full_df_llama, full_df_pegasus = plot_grade_level_distribution(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column)
    # full_df_human, full_df_gpt, full_df_llama, full_df_pegasus = complex_words_ratio(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column)
    # full_df_human, full_df_gpt, full_df_llama, full_df_pegasus = compute_dependency_tree_depth(full_df_human, full_df_gpt, full_df_llama, full_df_pegasus, column)

    # # Save the DataFrames to CSV files
    full_df_human.to_csv(f"{dataset_name}_full_df_human.csv")
    full_df_gpt.to_csv(f"{dataset_name}_full_df_gpt.csv")
    full_df_llama.to_csv(f"{dataset_name}_full_df_llama.csv")
    full_df_pegasus.to_csv(f"{dataset_name}_full_df_pegasus.csv")




    














# %%


# %%
