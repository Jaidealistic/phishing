# Phishing and Spam Detection using NLP and Transformer Models

This project explores and compares different machine learning approaches for detecting phishing URLs and spam emails. The goal is to build robust and accurate models to protect users from malicious content.

## Project Overview

The project focuses on two distinct datasets: one for email spam detection and another for URL phishing detection. We implement and evaluate traditional machine learning models (Logistic Regression and Random Forest with TF-IDF features) and fine-tune Transformer-based models (BERT for email and DistilBERT for URLs) for each task.

## Datasets

- **Email Dataset:** Contains email text and labels (ham/spam).
- **URL Dataset:** Contains URLs and various extracted features, including a status indicating whether the URL is legitimate or phishing.

The data loading and preprocessing steps are handled in the notebook, including splitting the data into training, validation, and testing sets.

## Methodology

1.  **Data Loading and Preprocessing:** Load the datasets and prepare them for model training. This includes handling categorical features and splitting the data.
2.  **Baseline Models:** Train and evaluate traditional models (Logistic Regression and Random Forest) using TF-IDF vectorized text features for both email and URL datasets.
3.  **Transformer Models:** Fine-tune pre-trained Transformer models (BERT for emails, DistilBERT for URLs) on the respective datasets.
4.  **Evaluation:** Evaluate all trained models on their respective test sets using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and ROC curves are generated to visualize performance.
5.  **Hyperparameter Tuning:** Perform basic hyperparameter sweeps for the Transformer models to identify potentially better configurations.
6.  **Analysis and Comparison:** Compare the performance of baseline models against the fine-tuned Transformer models and analyze the results.

## Results

The results of the model evaluations are summarized and visualized in the notebook and saved in the `outputs` folder. Key findings include:

-   Comparison of F1 scores across different models and datasets.
-   Confusion matrices for the best-performing models on each task.
-   Training history plots to assess model convergence and potential overfitting.

The fine-tuned Transformer models generally outperform the traditional baseline models on both tasks, achieving higher F1 scores.

## Code Structure

The project is implemented as a Google Colab notebook, with distinct sections for:

-   Setting up the environment and dependencies.
-   Loading and preprocessing the data.
-   Training and evaluating baseline models.
-   Preparing data for Transformer models.
-   Fine-tuning and evaluating Transformer models.
-   Comparing model performance and visualizing results.
-   Saving models and generating output files.

## Getting Started

To run this notebook:

1.  Open the notebook in Google Colab.
2.  Ensure you have a GPU runtime enabled (Runtime -> Change runtime type -> GPU).
3.  Upload the `email_data.csv` and `url_data.csv` files when prompted.
4.  Run all the cells in the notebook.

The output plots and tables will be saved in the `./outputs` directory.

## Dependencies

The project uses the following libraries:

-   `transformers`
-   `datasets`
-   `accelerate`
-   `evaluate`
-   `sentencepiece`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `imbalanced-learn`
-   `pandas`
-   `torch`
-   `numpy`

These dependencies are installed in the first cell of the notebook.

## Future Work

Possible extensions to this project include:

-   Exploring other Transformer architectures.
-   Implementing more advanced hyperparameter tuning techniques.
-   Investigating explainability methods to understand model predictions.
-   Deploying the trained models for real-time detection.
-   Handling imbalanced datasets with more sophisticated techniques.
