import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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
from annoy import AnnoyIndex
import torch
import gradio as gr
from PIL import Image
import requests
import io




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

def calculate_TFIDF_embeddings(df,tokenizer): #ici : text=df.overview.values
    tfidf = TfidfVectorizer(tokenizer=tokenizer)
    tfidf_matrix = tfidf.fit_transform(df.overview.values.astype('U')) 
    return tfidf_matrix, tfidf

def calculate_TFIDF_embeddings_user(tfidf_vectorizer,user_text):
    user_matrix = tfidf_vectorizer.transform([user_text])
    return user_matrix

def recommend_top_five_films_bagofwords(df,user_text):
    tokenizer=LemmatizerTokenizer()
    data_matrix,vectorizer=calculate_TFIDF_embeddings(df,tokenizer)
    user_matrix=calculate_TFIDF_embeddings_user(vectorizer,user_text)
    cosine_sim=cosine_similarity(data_matrix,user_matrix).flatten()
    top_indices = np.argsort(cosine_sim)[-5:][::-1]
    top_movies = df['overview'].iloc[top_indices]
    return top_movies

def get_df_distilbert_embeddings(path):
    df_distilbert = pd.read_pickle(path)
    return df_distilbert

def get_distilbert_embeddings_user(user_text):
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if pd.isna(user_text):
        return np.zeros(model.config.dim)
    model.to('cuda')
    inputs = tokenizer(user_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

def build_annoy_index(df_distilbert,user_embedding):
    dim_embedding=len(user_embedding)
    annoy_index=AnnoyIndex(dim_embedding,'angular')
    for i, embedding in enumerate(df_distilbert.distilbert_embeddings):
        annoy_index.add_item(i,embedding)
    annoy_index.build(n_trees=10)
    return annoy_index

def recommend_top_five_films_distilbert(df,user_text):
    path='distilbert_embeddings.pkl'
    df_distilbert=get_df_distilbert_embeddings(path)
    user_embedding=get_distilbert_embeddings_user(user_text)
    annoy_index=build_annoy_index(df_distilbert,user_embedding)
    indices = annoy_index.get_nns_by_vector(user_embedding, 5)
    top_movies = df.iloc[indices]
    return top_movies

def main(user_text, method):
    movie_dataset=pd.read_csv('movies_metadata.csv')
    df=movie_dataset[['id','title','release_date','overview']]
    if method == "Bag of Words":
        return recommend_top_five_films_bagofwords(df,user_text)
    elif method == "DistilBERT":
        return recommend_top_five_films_distilbert(df,user_text)
    else:
        return "Please select a method"

iface = gr.Interface(
    fn=main,
    inputs=[gr.inputs.Textbox(lines=2, placeholder="Enter movie description here..."), 
            gr.inputs.Radio(["Bag of Words", "DistilBERT"], label="Select Method")],
    outputs=[gr.outputs.Dataframe(type="pandas",label="Top 5 Movies")],
    description="Enter text to get top five movie recommendations using Bag of Words or DistilBERT methods."
)

iface.launch(debug=True, share=True)