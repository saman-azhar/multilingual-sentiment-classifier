## _Multilingual Sentiment Classifier_

A multilingual sentiment predictor which classifies English, Urdu or Roman Urdu text as Negative or Positive. It uses sentiment analysis algorithms of machine learning to classify negative and positive texts. We used 5 different algorithms and then calculated the sum of predictions to vote if a text is most likely to be negative or positive in sentiment.

## Features

- Multilingual Sentiment Analysis
- Preprocesses text before passing to classifiers
- Uses 5 different classifiers to vote if an input is positive or negative
- Sentiment Analysis of Urdu, Roman Urdu and English languages
- Can classify a single text
- Can also classify a DataFrame

## Requirements

This project uses a number of open source projects to work properly:

- [python 3.8]
- [nltk==3.6.2]
- [tweet-preprocessor==0.6.0]
- [textblob==0.15.3]
- [scikit-learn==0.22.2.post1]
- [numpy==1.18.5]
- [NLTK Data]

And of course this itself is open source with a [public repository][git-repo]
on GitHub.

## Installation

### Git:

```sh
git clone https://github.com/saman-azhar/multilingual-sentiment-classifier.git
```

### Pip:

```sh
pip3 install multilingualsentimentclassifier
```

## Module Example

### Text:

```sh
from multilingualsentimentclassifier.methods import text_sentiment
import nltk

#download these 2 dependencies when running for first time
nltk.download('punkt')
nltk.download('wordnet')

# testing english negative
sentiment = text_sentiment.predict_sentiment("i AM sad and angry :@", "en")
print(sentiment)

# testing roman-urdu postitive
sentiment2 = text_sentiment.predict_sentiment("main boht khush houn", "in")
print(sentiment2)

# testing urdu negative
# you can directly print in console
print(text_sentiment.predict_sentiment("میں تم سے ناراض ہوں", "ur"))
```

### DataFrame:

```sh
from multilingualsentimentclassifier.methods import dataframe_sentiment
import pandas as pd
import numpy as np

# dataframe w urdu text
df_ur = pd.DataFrame(np.array([["میں تم سے ناراض ہوں"], ["میں تم سے خوش ہوں"]]), columns=["text"])
print(dataframe_sentiment.predict_sentiment(df_ur, "ur"))
```

> Note: This software does not detect language. You must enter the language of your input data. 1 indicates negative and 0 indicates positive.

## License

MIT

[//]: # "These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax"
[python 3.8]: https://www.python.org/downloads/release/python-380/
[nltk==3.6.2]: https://pypi.org/project/nltk/
[tweet-preprocessor==0.6.0]: https://pypi.org/project/tweet-preprocessor/
[textblob==0.15.3]: https://pypi.org/project/textblob/
[scikit-learn==0.22.2.post1]: https://pypi.org/project/scikit-learn/0.22.2.post1/
[numpy==1.18.5]: https://pypi.org/project/numpy/1.18.5/
[nltk data]: https://www.nltk.org/data.html
[license]: https://github.com/saman-azhar/multilingual-sentiment-classifier/blob/main/LICENSE
[git-repo]: https://github.com/saman-azhar/multilingual-sentiment-classifier
