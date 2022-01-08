
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support import  expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import  time
from textblob import TextBlob
import plotly.express as px
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
from wordcloud import WordCloud
print(os.getcwd())
"""
Create dataframe of positive and negative  sentiment
"""

def create_dummy_data(csv_file):
    df_raw = pd.read_csv(csv_file)
    # New dataframe
    cols = ['Text', 'Sentiment']
    list_text = df_raw['selected_text'].tolist()
    list_sentiment = df_raw['sentiment'].tolist()
    list_new_data = list(zip(list_text, list_sentiment))
    df_new = pd.DataFrame(data=list_new_data, columns=cols)
    df_new = df_new.sort_values(by=['Sentiment'])

    # fill
    df_fill_pos = df_new[22376:22426]
    df_fill_neg = df_new[1:51]
    print(df_fill_neg[:5])
    # save_to csv
    df_fill_pos.to_csv('all_generated/dummy_positive.csv')
    df_fill_neg.to_csv('all_generated/dummy_negative.csv')
    print("Generate dummy data DONE!")
    return

def autofill_forms(formUrl, sentiment):
    driver = webdriver.Firefox(executable_path= r"E:\\__Talline_files\\Freelance\\geckodriver-v0.30.0-win64\\geckodriver.exe")
    formUrl = formUrl
    driver.get(formUrl)
    inputs = driver.find_element_by_class_name("quantumWizTextinputPaperinputInput")
    time.sleep(1)

    sentiment = sentiment
    inputs.clear()
    inputs.send_keys(sentiment)
    submit = driver.find_element_by_class_name("appsMaterialWizButtonPaperbuttonLabel")
    submit.click()
    WebDriverWait(driver,15).until(EC.visibility_of_element_located(
        (By.CLASS_NAME, "freebirdFormviewerViewResponseLinksContainer"
    )))
    another_response = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div[4]/a')
    another_response.click()
    # to refresh the browser
    driver.refresh()
    # to close the browser
    driver.close()
    return

#create_dummy_data('test/train.csv')
"""
autofill_forms("https://docs.google.com/forms/d/e/1FAIpQLSfQpYwVB2ROv14tvycVJ6fcnQ9pJT4MiMabTM_orDWm2ojW9Q/viewform",
               "I love my mom, she`s the best mom on the planet")
"""

def fill_from_file(pos_csv, neg_csv):
    df_pos = pd.read_csv(pos_csv)
    df_neg = pd.read_csv(neg_csv)

    list_pos = df_pos['Text'].tolist()
    list_neg = df_neg['Text'].tolist()

    for pos in list_pos:
        autofill_forms(
            "https://docs.google.com/forms/d/e/1FAIpQLSfQpYwVB2ROv14tvycVJ6fcnQ9pJT4MiMabTM_orDWm2ojW9Q/viewform",
            pos)

    for neg in list_neg:
        autofill_forms(
            "https://docs.google.com/forms/d/e/1FAIpQLSfQpYwVB2ROv14tvycVJ6fcnQ9pJT4MiMabTM_orDWm2ojW9Q/viewform",
            neg)
"""
fill_from_file("sentimentVis/all_generated/dummy_positive.csv",
               "sentimentVis/all_generated/dummy_negative.csv")
"""

def sentiment_analysis(csv_positive, csv_negative):
    df_positive = pd.read_csv(csv_positive)
    df_negative = pd.read_csv(csv_negative)

    list_positive = df_positive['Text'].tolist()
    list_negative = df_negative['Text'].tolist()

    cols = ["Text", "Polarity", "Subjectivity", "Predicted Sentiment", "True Sentiment"]
    data = []
    #subjectivity
    for p in list_positive:
        subjectivity = TextBlob(p).sentiment.subjectivity
        polarity = TextBlob(p).sentiment.polarity

        predicted_sentiment = "Positive"
        if polarity < 0:
            predicted_sentiment = "Negative"
        data.append([p, polarity, subjectivity, predicted_sentiment, "Positive"])
    for n in list_negative:
        subjectivity = TextBlob(n).sentiment.subjectivity
        polarity = TextBlob(n).sentiment.polarity
        predicted_sentiment = "Negative"
        if polarity > 0:
            predicted_sentiment = "Positive"
        data.append([n, polarity, subjectivity, predicted_sentiment, "Negative"])

    df_data = pd.DataFrame(data=data, columns=cols)
    print(df_data)
    return df_data

def visualize_sentiment(df):
    fig = px.scatter(df,
                     x='Polarity',
                     y='Subjectivity',
                     color='Predicted Sentiment',
                     size='Subjectivity',
                     hover_data=['Text', 'True Sentiment'])
    fig.update_layout(title='Sentiment Analysis',
                      shapes=[dict(type='line',
                                   yref='paper', y0=0, y1=1,
                                   xref='x', x0=0, x1=0)])
    fig.show()
    return

df = sentiment_analysis("sentimentVis/all_generated/dummy_positive.csv", "sentimentVis/all_generated/dummy_negative.csv")
#visualize_sentiment(df)

def wordcloud(df):
    return 