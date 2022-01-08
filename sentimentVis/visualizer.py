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
    plt.bar(x,y)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=fontsize, labelpad=10)
    plt.xticks(rotation=45)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_file)

def wordcloud(df, col_name, color, title, w, h, margin, min_font_size , figsize, output):
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
