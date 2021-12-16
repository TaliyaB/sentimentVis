
import os
import pandas as pd


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

def autofill_forms(google_form, df):
    return

create_dummy_data('test/train.csv')