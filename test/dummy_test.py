from sentimentVis import input_parser, text_processor , visualizer
from sklearn.feature_extraction.text import CountVectorizer

#input data
docs = input_parser.dummy_parse('sample_data/abcnews-date-text.csv')

r"""
Exploratory Data Analysis
Get Top N words, plot and save as PNG format
"""
top_n_words = text_processor.top_n_wordcounts(docs,20)
visualizer.plot_bar(
    x = top_n_words['Word'].tolist(),
    y = top_n_words['Frequency'].tolist(),
    fontsize =10,
    title = 'Top N Words',
    xlabel = 'Words',
    ylabel = 'Frequency',
    output_file = 'all_generated/plot.png'
)

r"""
Perform Sentiment Analysis and organize into two dataframes.
"""
positive, negative = text_processor.sentiment_analysis(
    df = docs,
    col_name = 'headline_text',
    positive_polarity_floor= 0.01,
    positive_subjectivity_floor = 0.4
)

r"""
Generate WordCloud for Positive and Negative Sentiments
"""
visualizer.wordcloud(
    df = positive,
    col_name = 'text',
    color='white',
    title = 'Predicted Positive Sentiments',
    w=1600,
    h=800,
    margin=2,
    min_font_size=20,
    figsize=(15,10),
    output='all_generated/positive_wordcloud.png'
)

visualizer.wordcloud(
    df = negative,
    col_name = 'text',
    color='black',
    title = 'Predicted Negative Sentiments',
    w=1600,
    h=800,
    margin=2,
    min_font_size=20,
    figsize=(15,10),
    output='all_generated/negative_wordcloud.png'
)

r"""
Topic Modeling Using Latent Dirichlet Allocation
"""