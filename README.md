# Fake-News-Detection-After-LLM-Laundering

With their advanced capabilities, Large Language Models (LLMs) can generate highly convincing and contextually relevant fake news, which can contribute to disseminating misinformation. Though there is much research on fake news detection for human-written text, the field of detecting LLM-generated fake news is still under-explored. This research measures the efficacy of detectors in identifying LLM-paraphrased fake news, in particular, determining whether adding a paraphrase step in the detection pipeline helps or impedes detection. This study contributes: (1) Detectors struggle to detect LLM-paraphrased fake news more than human-written text, (2) We find which models excel at which tasks (evading detection, paraphrasing to evade detection, and paraphrasing for semantic similarity). (3) Via LIME explanations, we discovered a possible reason for detection failures: sentiment shift. (4) We discover a worrisome trend for paraphrase quality measurement: samples that exhibit sentiment shift despite a high BERTSCORE. (5) We provide a pair of datasets augmenting existing datasets with paraphrase outputs and scores.



- **llama_fine-tune.py:** Contains the code to fine-tune a Llama model for text classification
- **Paraphrasing_using_PLMs.py:** Code to generate paraphrase text
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

