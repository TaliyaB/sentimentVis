"""
Contains the code for Exploratory Data Analysis
 And machine learning processes
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from textblob import TextBlob

def top_n_wordcounts(df, n):
    txt=df.headline_text.str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(txt)
    words_dist = nltk.FreqDist(words)

    stop_words = nltk.corpus.stopwords.words('english')

    words_except_stop_dist = nltk.FreqDist(w for w in words if w not  in stop_words)

    print("All frequencies, including STOPWORDS: ")
    print('='*60)
    res = pd.DataFrame(words_except_stop_dist.most_common(n), columns=['Word', 'Frequency'])
    print(res)
    return  res

def exploratory_data_analysis(df):
    return df

def sentiment_analysis(df, col_name, positive_polarity_floor,
                       positive_subjectivity_floor):
    data = df[col_name].tolist()
    print("Sentiment_Analysis")
    print('=' * 60)
    print(data[:10])

    positive_sentiments = []
    negative_sentiments = []

    positive_polarity_scores = []
    negative_polarity_scores = []

    positive_subjectivity_scores = []
    negative_subjectivity_scores = []

    for i in range(len(data)):
        blob = TextBlob(data[i])

        #conditions for predicting
        if (blob.sentiment.polarity >= positive_polarity_floor) and (blob.sentiment.subjectivity>=positive_subjectivity_floor):
            positive_sentiments.append(data[i])
            positive_polarity_scores.append(blob.sentiment.polarity)
            positive_subjectivity_scores.append(blob.sentiment.subjectivity)
        else:
            negative_sentiments.append(data[i])
            negative_polarity_scores.append(blob.sentiment.polarity)
            negative_subjectivity_scores.append(blob.sentiment.subjectivity)

    #create dataframe
    data_cols = ['text', 'polarity' , 'subjectivity']
    df_positive_preds = pd.DataFrame(list(zip(positive_sentiments, positive_polarity_scores, positive_subjectivity_scores)), columns = data_cols)
    df_negative_preds = pd.DataFrame(list(zip(negative_sentiments, negative_polarity_scores, negative_subjectivity_scores)), columns = data_cols)
    return (df_positive_preds, df_negative_preds)

def latent_dirichlet_allocation(df):
    return df
