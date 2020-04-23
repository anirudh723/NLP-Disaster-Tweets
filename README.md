# Real or Not? NLP with Disaster Tweets

#### -- Project Status: Completed

## Project Intro/Objective
Twitter can be a valuable mechaism for conveying important information. If the intended meaning behind a tweet can be interpreted properly,
it can saves lives in time of a disaster. 
The purpose of this project is to use supervised machine learning techniques in order to predict the disaster status of a given tweet,
based on a training subset of a collection of twitter data. 
This project is viewed as a sentiment analysis natural language processing (NLP) problem. 

This work holds significance because it advances our ability to respond appropriately to situations through the vast amount of information
available to us, especially through social media platforms, such as Twitter.

### Methods Used
* Machine Learning
* NLP Techniques - Sentiment Analysis
* Classification Modeling
* Data Visualization
* Inferential Statistics

### Technologies
* Python
* Jupyter Notebook
* Python libaries - scikit-learn, pandas, numpy, matplotlib, seaborn

## Project Description

This project is based from Kaggle, which is an online community of data scientists, on their 'Real or Not? NLP with Disaster Tweets'
competition. The training data set, called train.csv, which was provided by Kaggle to allow machine learning techniques to be performed on,
is provided in the link below. It was renamed to data.csv in this project.

https://www.kaggle.com/c/nlp-getting-started/data

In this project, the best possible model for labeling the disaster status of tweets is sought to be created. 
The questions to be asked:
* How effective is Multinomial Naive Bayes compared to other classification algorithms?
    * It is expected to be more effective than the alternatives, as it is a popular algorithm for text classification, including sentiment analysis.
* What parameters for TfidVectorizer are best for vocabulary construction?
    * Which set of parameters best reduces the risk of overfitting to the training data while maintaining good performance?
    * Does the use of n-grams increase accuracy/precision significantly when formulating a test model with data that is so prone to contain sarcasm?

The following hypotheses were formed:
* Multinomial Naive Bayes will be the most effective classification algorithm.
* The use of n-grams will give a noticeable increase to accuracy/precision because our NLP model is so prone to miscontextualization.

Steps to prove/disprove hypotheses:
* Import dataset, extract features and target column variables
    * Feature columns consist of text (tweets), keywords, location, etc. Target column was a binary representation
    of whether the tweet from the dataset had represented a disaster (1 for real disaster, 0 for fake disaster)
    * The non-text features (keyword, location) were passed through a preprocessing pipeline, where NaN values were imputed to be a missing value variable.
    The categorical variables were then hot encoded. 
    * Further data cleaning occured by removing punctuation from the tweets
    * Dataset was split into training and testing subsets 
    * Different TfidfVectorizers were tested on the training set, using different params, including n-grams, stop-words, etc. F1 scores were used to determine
    the best params for vocabulary construction
    * Three different classification algorithms were tested (Multinomial Naive Bayes, Logistic Regression, and Decision Tree)
    For consistency and to get ideal hyperparameters, GridSearchCV was used. Logistic Regression had the highest F1 Score by a slight margin.
    * Hypothesis tests were also ran to see if any of the classifiers were significantly better than the others. 
    Logistic Regression and MultinomialNB were not statistically signficant from each other, but DecisionTree was significantly worse.

## Getting Started

1. Clone this repo (for help see this (https://help.github.com/articles/cloning-a-repository/)).
2. Make sure Jupyter Notebook and raw data files are kept in the same directory
5. Follow the notebook, which contains code, annotations, in-depth descriptions, etc