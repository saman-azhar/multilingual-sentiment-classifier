from multilingualsentimentclassifier.methods import text_sentiment, dataframe_sentiment
import pandas as pd
import numpy as np


def test_predict_sentiment_eng():
    # testing english negative
    assert text_sentiment.predict_sentiment(
        "i AM sad and angry :@", "en") == 'Negative'
    # testing english positive
    assert text_sentiment.predict_sentiment(
        "i am happy and Over The MOON!", "en") == 'Positive'


def test_predict_sentiment_romur():
    # testing roman-urdu negative
    assert text_sentiment.predict_sentiment(
        "sab boht fazool hain", "in") == 'Negative'
    # testing roman-urdu postitive
    assert text_sentiment.predict_sentiment(
        "main boht khush houn", "in") == 'Positive'


def test_predict_sentiment_urdu():
    # testing urdu negative
    assert text_sentiment.predict_sentiment(
        "میں تم سے ناراض ہوں", "ur") == 'Negative'
    # testing urdu positive
    assert text_sentiment.predict_sentiment(
        "میں تم سے خوش ہوں", "ur") == 'Positive'


def test_predict_sentiment_eng_df():
    # dataframe w english text
    df_eng = pd.DataFrame(
        np.array([["i am happy"], ["i am SAD"]]), columns=["text"])
    assert dataframe_sentiment.predict_sentiment(df_eng, "en").equals(pd.DataFrame(
        {'text': ["i am happy", "i am sad"], 'preprocessed': ["happy", "sad"], 'sentiment': [0, 1]}))


def test_predict_sentiment_urdu_df():
    # dataframe w urdu text
    df_ur = pd.DataFrame(
        np.array([["میں تم سے ناراض ہوں"], ["میں تم سے خوش ہوں"]]), columns=["text"])
    assert dataframe_sentiment.predict_sentiment(df_ur, "ur").equals(pd.DataFrame(
        {'text': ["میں تم سے ناراض ہوں", "میں تم سے خوش ہوں"], 'preprocessed': ["میں تم سے ناراض ہوں", "میں تم سے خوش ہوں"], 'sentiment': [1, 0]}))


def test_predict_sentiment_romur_df():
    # dataframe w roman urdu text
    df_in = pd.DataFrame(
        np.array([["iski quality itni fazool hay"], ["main boht KHUSH houn"]]), columns=["text"])
    assert dataframe_sentiment.predict_sentiment(df_in, "in").equals(pd.DataFrame(
        {'text': ["iski quality itni fazool hay", "main boht khush houn"], 'preprocessed': ["iski quality itni fazool", "boht khush"], 'sentiment': [1, 0]}))
