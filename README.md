# Sentiment and Emotion Analysis in User Reviews

## Overview

This repository contains experimental work and datasets for analyzing sentiment and emotion in user reviews. The project explores deep learning models like GRU, BiGRU, LSTM, BiLSTM, CNN, and RNN, and includes sentiment analysis using VADER. The datasets have been annotated both manually and by ChatGPT. The repository provides Jupyter notebooks for model implementation and experimentation, as well as theoretical discussions in PDF format.

## Directory Structure

### Code Files
- **Algorithm 1 Processing Reviews for Sentiment.ipynb**: Preprocesses user reviews to prepare data for sentiment analysis.

- **BIGRU_ChatGPT annotatted datset.ipynb** & **BIGRU__manual_annotated_dataset.ipynb**: Implement BiGRU models for ChatGPT-annotated and manually annotated datasets, respectively, to compare model performance across annotation methods.

- **CNN Model for chatgpt annotated dataset.ipynb** & **CNN for Mnaul annotatted dataset.ipynb**: Develop CNN models for analyzing ChatGPT and manually annotated datasets separately, comparing performance across annotation techniques.

- **GRU_Chatgpt_dataset_model.ipynb** & **GRU_Manual_dataset_model.ipynb**: Implement GRU models for ChatGPT and manually annotated datasets, with performance comparisons.

- **Grid search for(chatgptdataset).ipynb**: Hyperparameter tuning for the ChatGPT-annotated dataset, optimizing parameters like learning rate and batch size.

- **LSTM, BILSTM, RNN for ChatGPt Dataset.ipynb** & **LSTM,BILSTM for Manual Dataset.ipynb**: Implement LSTM, BiLSTM, and RNN models for sentiment analysis on ChatGPT and manually annotated datasets, evaluating performance using metrics such as precision, recall, and F1-score.

- **RNN Model_chatgpt_annotated.ipynb** & **RNN Model_manual_annotated_dataset.ipynb**: Build RNN models for sentiment analysis on both ChatGPT and manually annotated datasets.

- **Vadar setiment anylysis.ipynb**: Uses VADER for sentiment analysis, providing polarity scores and sentiment classifications.

### PDF Files
- **Understanding Emotions in End.pdf**: Theoretical discussion on understanding emotions in user reviews, offering insights into emotion detection strategies.

### Datasets
- **Dataset_chatgpt_manul_final.csv**: A combined dataset containing ChatGPT and manually annotated sentiment labels.

- **end-user reviews dataset.csv**: End-user reviews labeled with sentiment and emotion annotations.

- **vadar_resutls.csv**: Results generated from applying VADER sentiment analysis.

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
   - Load datasets such as `Dataset_chatgpt_manul_final.csv` and `end-user reviews dataset.csv` to understand their structure.
   - Visualize data distributions using tools like `matplotlib` or `seaborn`.

2. **Perform Sentiment Analysis**:
   - Use **Vadar setiment anylysis.ipynb** for VADER-based sentiment analysis.
   - Experiment with deep learning models like GRU, CNN, or LSTM in notebooks such as **GRU_Chatgpt_dataset_model.ipynb**.

3. **Optimize Hyperparameters**:
   - Use **Grid search for(chatgptdataset).ipynb** to tune model parameters for improved accuracy.

4. **Compare Models**:
   - Evaluate model performance (e.g., CNN, RNN, BiGRU, BiLSTM) using metrics like accuracy and confusion matrices.
   - Analyze how annotation methods (manual vs. ChatGPT) affect model performance.

5. **Utilize PDF Resources**:
   - Refer to **Understanding Emotions in End.pdf** for theoretical insights and additional analysis techniques.

## Contact Information

For questions or feedback regarding this project, please contact [naikdil2003@gmail.com].
