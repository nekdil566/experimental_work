# Sentiment and Emotion Analysis in User Reviews

## Overview

This repository contains the experimental work and datasets for analyzing sentiment and emotion in user reviews. The project explores various deep learning models such as GRU, BiGRU, LSTM, BiLSTM, CNN, and RNN. Additionally, sentiment analysis using VADER is implemented. The datasets include annotations from both ChatGPT and manual labeling processes. This repository provides Jupyter notebooks for implementation and experimentation, alongside theoretical discussions in PDF format.

## Directory Structure

### Code Files
- **Algorithm 1 Processing Reviews for Sentiment.ipynb**:
  Notebook focused on the preprocessing of user reviews to prepare the data for sentiment analysis.

- **GRU, BiGRU, Chatgpt, Manual_annoated_ dataset.ipynb**:
  Implements GRU and BiGRU models for sentiment analysis. Compares performance on ChatGPT-annotated and manually annotated datasets.

- **CNN Model for chatgpt annotated dataset.ipynb**:
  Develops a CNN model for analyzing ChatGPT-annotated datasets.

- **CNN for Manual Annotated dataset.ipynb**:
  Implements a CNN model specifically for manually annotated datasets. Provides performance comparison with ChatGPT-annotated datasets.

- **Grid search for(chatgptdataset).ipynb**:
  Performs hyperparameter tuning using grid search for ChatGPT datasets. Optimizes parameters such as learning rate and batch size.

- **Grid search For GRU.ipynb**:
  Optimizes GRU model parameters through grid search.

- **LSTM, BILSTM, RNN for ChatGPt Dataset.ipynb**:
  Implements LSTM, BiLSTM, and RNN models for sentiment analysis on ChatGPT datasets. Evaluates their performance with metrics such as precision, recall, and F1-score.

- **LSTM, BILSTM for Manual Dataset.ipynb**:
  Focuses on LSTM and BiLSTM models for manually annotated datasets. Addresses class imbalance using sampling techniques.

- **RNN_Manual_data.ipynb**:
  Builds RNN models for sentiment analysis on manually annotated datasets.

- **Vadar setiment anylysis.ipynb**:
  Implements VADER sentiment analysis on the datasets. Outputs include polarity scores and sentiment classifications.

### PDF Files
- **Understanding Emotions in End.pdf**:
  Theoretical discussion on understanding emotions in user reviews. Provides insights into emotion detection strategies.

### Datasets
- **Dataset_chatgpt_manul_final.csv**:
  A combined dataset with ChatGPT-annotated and manually annotated sentiment labels.

- **end-user reviews dataset.csv**:
  Contains end-user reviews labeled with sentiment and emotion annotations.

- **vadar_resutls.csv**:
  Results generated from applying VADER sentiment analysis.

## Requirements

To run the code provided in this repository, you will need:

- **Python 3.x**: The programming language used in this project.
- **Jupyter Notebook or JupyterLab**: For running `.ipynb` files.
- **Python Libraries**:
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
  - `matplotlib` or `seaborn`: For data visualization.
  - `scikit-learn`: For machine learning tasks.
  - `keras` and `tensorflow`: For designing and training deep learning models.
  - `nltk` or `vaderSentiment`: For natural language processing and sentiment analysis.

## Running the Analysis

1. **Explore the Datasets**:
   - Load the datasets such as `Dataset_chatgpt_manul_final.csv` and `end-user reviews dataset.csv` to understand their structure.
   - Visualize data distributions using tools like `matplotlib` or `seaborn`.

2. **Perform Sentiment Analysis**:
   - Use the **Vadar setiment anylysis.ipynb** notebook for basic sentiment analysis using VADER.
   - Experiment with deep learning models such as GRU, CNN, or LSTM in notebooks like **GRU, BiGRU, Chatgpt, Manual_annoated_ dataset.ipynb**.

3. **Optimize Hyperparameters**:
   - Use **Grid search for(chatgptdataset).ipynb** or **Grid search For GRU.ipynb** to tune model parameters for better accuracy.

4. **Compare Models**:
   - Evaluate the performance of various models (e.g., CNN, RNN, BiGRU, BiLSTM) using metrics like accuracy and confusion matrices.
   - Analyze the impact of annotation techniques (manual vs. ChatGPT) on model performance.

5. **Leverage PDF Resources**:
   - Theoretical insights and reusable code snippets are available in PDFs like **Understanding Emotions in End.pdf**.

## Contact Information

For questions or feedback regarding this project, please contact [naikdil2003@gmail.com].

