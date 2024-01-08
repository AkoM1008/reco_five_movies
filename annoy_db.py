import pandas as pd
import numpy as np 
import unicodedata
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import pickle

def get_df_distilbert_embeddings(path):
    df_distilbert = pd.read_pickle(path)
    if not isinstance(df_distilbert, pd.DataFrame):
        raise ValueError("Loaded object is not a DataFrame")
    return df_distilbert

def get_distilbert_embeddings_user(user_text):
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if pd.isna(user_text):
        return np.zeros(model.config.dim)
    #model.to('cuda')
    inputs = tokenizer(user_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    #inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

def recommend_top_five_films_distilbert(df,user_text):
    path='distilbert_embeddings.ann'
    df_distilbert=get_df_distilbert_embeddings(path)
    user_embedding=get_distilbert_embeddings_user(user_text)
    indices = df_distilbert.get_nns_by_vector(user_embedding, 5)
    top_movies = df.iloc[indices]
    return top_movies

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class LemmatizerTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_wordnet_pos(self, word, pos_tag):
        tag = pos_tag[0].upper()
        tag_dict = {"J": ADJ, "N": NOUN, "V": VERB, "R": ADV}
        return tag_dict.get(tag, NOUN)

    def __call__(self, doc):
        doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8')
        tokens = word_tokenize(doc.lower())
        pos_tags = pos_tag(tokens)
        return [self.lemmatizer.lemmatize(t, self.get_wordnet_pos(t, pos)) for t, pos in pos_tags if t not in self.stop_words and t.isalpha()]


def get_TFIDF_embeddings(path): #ici : text=df.overview.values
    df_tfidf = pd.read_pickle(path)
    if not isinstance(df_tfidf, pd.DataFrame):
        raise ValueError("Loaded object is not a DataFrame")
    return df_tfidf

def calculate_TFIDF_embeddings_user(tfidf_vectorizer,user_text):
    user_matrix = tfidf_vectorizer.transform([user_text])
    return user_matrix

def recommend_top_five_films_bagofwords(df,user_text):
    path_vectorizer='tokenizer_instance.pkl'
    with open(path_vectorizer,'rb') as file:
        vectorizer=pickle.load(file)
    path='tfidf_embeddings.ann'
    data_matrix=get_TFIDF_embeddings(path)
    user_matrix=calculate_TFIDF_embeddings_user(vectorizer,user_text)
    cosine_sim=cosine_similarity(data_matrix,user_matrix).flatten()
    top_indices = np.argsort(cosine_sim)[-5:][::-1]
    top_movies = df.iloc[top_indices]
    return pd.DataFrame(top_movies)