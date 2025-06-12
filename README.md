# Fake-News-Detection-After-LLM-Laundering

With their advanced capabilities, Large Language Models (LLMs) can generate highly convincing and contextually relevant fake news, which can contribute to disseminating misinformation. Though there is much research on fake news detection for human-written text, the field of detecting LLM-generated fake news is still under-explored. This research measures the efficacy of detectors in identifying LLM-paraphrased fake news, in particular, determining whether adding a paraphrase step in the detection pipeline helps or impedes detection. This study contributes: (1) Detectors struggle to detect LLM-paraphrased fake news more than human-written text, (2) We find which models excel at which tasks (evading detection, paraphrasing to evade detection, and paraphrasing for semantic similarity). (3) Via LIME explanations, we discovered a possible reason for detection failures: sentiment shift. (4) We discover a worrisome trend for paraphrase quality measurement: samples that exhibit sentiment shift despite a high BERTSCORE. (5) We provide a pair of datasets, augmenting existing datasets with paraphrase outputs and scores.

**Coding Description**

- **llama_fine-tune.py:** Contains the code to fine-tune a Llama model for text classification
- **Paraphrasing_using_PLMs.py:** Code to generate paraphrased text
- **sentiment analysis.py:** Code for sentiment analysis
- **linguistic_features.py:** Code for analyzing linguistic features
- **paraphrase_evaluation.py:** Code for evaluating paraphrasing text.
- **deep_learning.py:** Code for deep learning classification techniques
- **gpt_fine-tune_2.py:** Contains the code to fine-tune a GPT model for text classification
- **load_data.py:** Load the datasets
- **supervised_learning_2.py:** Code for supervised classification techniques
- **bert_classifier:** Contains the code to fine-tune a BERT model for text classification
- **T5_classifier:** Contains the code to fine-tune a T5 model for text classification
- **explainer_cnn_lstm:** Code for the LIME explainability

**Results**
The classification performance for the human-written versus LLM paraphrased Covid-19 dataset is shown below.
|           | Human-written | GPT-generated | Llama-generated | Pegasus-generated |
| --------- | ------------- | ------------- | --------------- | ----------------- |
| Pipeline  | Acc           | F1            | Pre             | Rec               | Acc | F1 | Pre | Rec | Acc | F1 | Pre | Rec | Acc | F1 | Pre | Rec |
| BERT      | 0.93          | 0.93          | 0.93            | 0.93              | 0.922 | 0.922 | 0.922 | 0.922 | 0.902 | 0.902 | 0.902 | 0.902 | 0.877 | 0.877 | 0.877 | 0.877 |
| T5        | 0.93          | 0.932         | 0.94            | 0.93              | 0.899 | 0.899 | 0.901 | 0.899 | 0.904 | 0.904 | 0.904 | 0.904 | 0.868 | 0.868 | 0.871 | 0.868 |
| Llama     | 0.939         | 0.939         | 0.94            | 0.939             | 0.918 | 0.918 | 0.918 | 0.918 | 0.927 | 0.927 | 0.927 | 0.927 | 0.879 | 0.879 | 0.879 | 0.879 |
| GPT-2     | 0.979         | 0.979         | 0.979           | 0.979             | 0.938 | 0.938 | 0.939 | 0.938 | 0.925 | 0.925 | 0.925 | 0.925 | 0.88 | 0.88 | 0.88 | 0.88 |
| CNN       | 0.92          | 0.92          | 0.92            | 0.92              | 0.903 | 0.903 | 0.903 | 0.903 | 0.887 | 0.887 | 0.887 | 0.887 | 0.852 | 0.852 | 0.852 | 0.852 |
| LSTM      | 0.924         | 0.924         | 0.924           | 0.924             | 0.906 | 0.906 | 0.906 | 0.906 | 0.895 | 0.895 | 0.895 | 0.895 | 0.868 | 0.868 | 0.868 | 0.868 |
| SVM-cv    | 0.914         | 0.914         | 0.914           | 0.914             | 0.891 | 0.891 | 0.891 | 0.891 | 0.88 | 0.88 | 0.88 | 0.88 | 0.858 | 0.858 | 0.859 | 0.858 |
| SVM-tfidf | 0.921         | 0.921         | 0.921           | 0.921             | 0.908 | 0.908 | 0.908 | 0.908 | 0.896 | 0.896 | 0.896 | 0.896 | 0.864 | 0.864 | 0.864 | 0.864 |
| SVM-wv    | 0.874         | 0.874         | 0.874           | 0.874             | 0.866 | 0.866 | 0.866 | 0.866 | 0.854 | 0.854 | 0.854 | 0.854 | 0.84 | 0.84 | 0.841 | 0.84 |
| LR-cv     | 0.921         | 0.921         | 0.921           | 0.921             | 0.902 | 0.902 | 0.903 | 0.902 | 0.893 | 0.893 | 0.893 | 0.893 | 0.866 | 0.866 | 0.866 | 0.866 |
| LR-tfidf  | 0.913         | 0.913         | 0.914           | 0.913             | 0.899 | 0.899 | 0.9 | 0.899 | 0.89 | 0.89 | 0.89 | 0.89 | 0.863 | 0.863 | 0.863 | 0.863 |
| LR-wv     | 0.868         | 0.868         | 0.868           | 0.868             | 0.86 | 0.86 | 0.86 | 0.86 | 0.852 | 0.852 | 0.852 | 0.852 | 0.837 | 0.837 | 0.838 | 0.837 |
| RF-cv     | 0.9           | 0.9           | 0.9             | 0.9               | 0.886 | 0.886 | 0.886 | 0.886 | 0.877 | 0.877 | 0.877 | 0.877 | 0.852 | 0.852 | 0.855 | 0.852 |
| RF-tfidf  | 0.899         | 0.899         | 0.9             | 0.899             | 0.885 | 0.885 | 0.885 | 0.885 | 0.871 | 0.871 | 0.871 | 0.871 | 0.851 | 0.851 | 0.852 | 0.851 |
| RF-wv     | 0.868         | 0.868         | 0.871           | 0.868             | 0.85 | 0.85 | 0.852 | 0.85 | 0.835 | 0.835 | 0.837 | 0.835 | 0.825 | 0.825 | 0.828 | 0.825 |
| DT-cv     | 0.856         | 0.855         | 0.855           | 0.856             | 0.807 | 0.807 | 0.807 | 0.807 | 0.793 | 0.793 | 0.793 | 0.793 | 0.804 | 0.804 | 0.804 | 0.804 |
| DT-tfidf  | 0.846         | 0.846         | 0.846           | 0.846             | 0.807 | 0.806 | 0.808 | 0.807 | 0.785 | 0.785 | 0.785 | 0.785 | 0.786 | 0.786 | 0.786 | 0.786 |
| DT-wv     | 0.77          | 0.77          | 0.77            | 0.77              | 0.742 | 0.742 | 0.742 | 0.742 | 0.735 | 0.735 | 0.737 | 0.735 | 0.721 | 0.722 | 0.723 | 0.721 |
