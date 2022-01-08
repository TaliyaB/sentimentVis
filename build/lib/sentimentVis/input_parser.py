"""
Contains the input parser for Google Forms data.
"""
import  pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def parse(csv):
    return  csv

def dummy_parse(csv):
    dummy_df = pd.read_csv(csv, usecols=['headline_text'])
    return dummy_df
"""
Get a list of top words used across the data.
"""
