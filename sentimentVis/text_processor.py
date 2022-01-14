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
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
spacy.cli.download("en_core_web_md")
import en_core_web_md

def top_n_wordcounts(df, n):
    print("Preview of Top {} Words".format(n))
    print("="*60)
    txt=df.headline_text.str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(txt)
    words_dist = nltk.FreqDist(words)
    stop_words = nltk.corpus.stopwords.words('english')
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not  in stop_words)
    res = pd.DataFrame(words_except_stop_dist.most_common(n), columns=['Word', 'Frequency'])
    print(res.sample(5))
    return  res

def exploratory_data_analysis(df):
    return df

def clean_texts(txts):
    return txts

def lemmatization(txts):
    allowed_postags=['NOUN','ADJ']
    nlp = en_core_web_md.load(disable=['parser', 'ner'])
    output = []
    for txt in txts:
        doc = nlp(txt)
        output.append([token.lemma_
                      for token in doc if token.pos_ in allowed_postags])
    return output

def sentiment_analysis(df, col_name, positive_polarity_floor,
                       positive_subjectivity_floor):
    print("Preview of Sentiment Analysis")
    print("="*60)
    data = df[col_name].tolist()
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
    print("PREDICTED POSITIVE SENTIMENTS")
    print("PREDICTED NEGATIVE SENTIMENTS")
    return (df_positive_preds, df_negative_preds)

def latent_dirichlet_allocation(df, col_name, output_graph):
    #raw text
    text_list = df[col_name].tolist()

    #convert to vector
    tf_vectorizer = CountVectorizer(
        strip_accents = 'unicode',
        stop_words = 'english',
        lowercase =True,
        token_pattern = r'\b[a-zA-Z]{3,}\b' , #num chars > 3 to avoid meaningless words
        max_df = 0.9, #remove words that appear 90% of the time
        min_df = 10 #discard terms that appear <10
    )

    #apply transformation
    tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
    print("TFIDF Params: ", tfidf_vectorizer)

    #convert to Document-Term Matrix
    dtm_tfidf = tfidf_vectorizer.fit_transform(text_list)
    print("The shape of the tfidf is {}, meaning that there are {} {} and {} tokens made through the filtering process.".format(dtm_tfidf.shape,dtm_tfidf.shape[0], col_name, dtm_tfidf.shape[1]))

    # GRID SEARCH & Parameter Tuning  to find optimal LDA model

    #search param
    search_params = {
        'n_components':[5,10, 15, 20, 25, 30],
        'learning_decay': [.5 ,.7 , .9]
    }

    #init model
    lda = LatentDirichletAllocation()
    model = GridSearchCV(lda, param_grid = search_params)

    #grid search
    model.fit(dtm_tfidf)

    # OUTPUT OPTIMAL MODEL
    best_lda_model = model.best_estimator_
    print("Model best parameters: {}".format(model.best_params_))
    print("Model Likelihood Score: {}".format(model.best_score_))
    # Perplexity , lower the better
    print("Model Perplexity: {}".format(best_lda_model.perplexity(dtm_tfidf)))

    # Compare LDA model Performance Scores
    # Get log likelihoods from Grid Search
    gscore = model.fit(dtm_tfidf).cv_results_
    n_topics = [5 ,10, 15, 20, 25, 30]

    log_likelyhoods_5 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.5]
    log_likelyhoods_7 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.7]
    log_likelyhoods_9 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.9]

    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, log_likelyhoods_5, label='0.5')
    plt.plot(n_topics, log_likelyhoods_7, label='0.7')
    plt.plot(n_topics, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.savefig(output_graph)

    return best_lda_model, dtm_tfidf, tfidf_vectorizer


