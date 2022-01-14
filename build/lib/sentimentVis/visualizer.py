"""
Contains the Visualizer
"""
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import  pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer

def visualize(csv):
    return csv

def plot_bar(x,y, fontsize, title, xlabel, ylabel, output_file):
    print("Generating {} Bar".format(title))
    plt.bar(x,y)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=fontsize, labelpad=10)
    plt.xticks(rotation=45)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.savefig(output_file)

def display_img(img):
    img = plt.imread(img)
    plt.imshow(img)
    plt.show()

def wordcloud(df, col_name, color, title, w, h, margin, min_font_size , figsize, output):
    print("Generating {} WordCloud...".format(title))
    text = df[col_name].tolist()
    text_str = ''
    stopwords = set(STOPWORDS)

    for txt in text:
        tokens = txt.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        text_str+=' '.join(tokens)+' '
    wordcloud = WordCloud(
        collocations = False,
        background_color = color,
        stopwords = stopwords,
        width = w,
        height = h,
        margin = margin,
        min_font_size = min_font_size
    ).generate(text_str)

    plt.figure(figsize = figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.figtext(0.5, 0.8, title, fontsize=20, ha='center')
    plt.savefig(output)
    plt.show()


def lda_visual(model, feature_names, n_top_words):
    topic_dict = {}
    for topic_idx , topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx+1)] = ["{}".format(feature_names[i])
                        for i in topic.argsort()[:-n_top_words-1 :-1]]
        topic_dict["Topic %d weights" % (topic_idx+1)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]
    df_topic = pd.DataFrame(topic_dict)
    print(df_topic)
    return df_topic

