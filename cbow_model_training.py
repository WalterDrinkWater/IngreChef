import pandas as pd
from gensim.models import Word2Vec
from preprocessing import ingredient_parser,get_and_sort_corpus
import config
import nltk

def get_window(corpus):
    lengths = [len(doc) for doc in corpus]
    avg_len = float(sum(lengths)) / len(lengths)
    return round(avg_len)

nltk.download('wordnet')
data = pd.read_csv(config.RECIPES_DETAILS)
data['parsed'] = data.ingredients.apply(ingredient_parser)
corpus = get_and_sort_corpus(data)

model_cbow = Word2Vec(corpus, sg=0, workers=8, window=get_window(corpus), min_count=1, vector_size=1000)
model_cbow.save(config.MODEL)
print("Word2Vec model successfully trained")