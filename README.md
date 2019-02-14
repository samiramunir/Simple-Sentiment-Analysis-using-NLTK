# Simple-Sentiment-Analysis-using-NLTK

### Introduction: 
This project aims to perform a simple binary classification of IMDB movie reviews as positive or negative. 

### Data: 
I used the IMDB movie reviews dataset from Kaggle which has 50000 reviews. Description of the data can be found at the following link. 
https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset. 

### Process: 

For the project I aimed to replicate the steps listed in this page for sentiment analysis of tweets. https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/

The Algorithm : 
Tokenize, clean and lemmatize the data and took only the adjectives from the reviews. 
Created a frequency distribution and found the most used words in all of the reviews. The top 5000 was used as the features.  

Make features vectors:
Created a dictionary representation for each review of weather each top word (mentioned above) exists in the review or not. The key in the dictoinary was each of the top words and the corresponding value was True of False for weather 'Word was in the review or not' 
Divdided the data into train test split. 
Use 7 different classification models to train on the data. Namely: 

Classifiers for an ensemble model from NLTK-ScikitLearn library: 
Naive Bayes (NB)
Multinomial NB
Bernoulli NB
Logistic Regression
Stochastic Gradient Descent Classifier - SGD
Support Vector Classification - SVC
Linear SVC
Nu- NuSVC

The output of the basic NB Classifier: 

![](Screen\ Shot\ 2019-02-13\ at\ 6.25.47\ PM.png)

Ensemple.py:  Created a classifier class that takes votes from each classifier and takes the mode of the votes to give the final decision with a certain confidence level.
Module.py: Can predict the sentiment of a single review. 







