import time
import pandas as pd
import plotly.express as px
import umap
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import hdbscan
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import requests
import seaborn as sns
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem import PorterStemmer
from utility_funtions import *
from nltk.stem.porter import *
import numpy as np
import nltk
from gensim import corpora, models
import plotly.graph_objects as go
from plotly.subplots import make_subplots
nltk.download('wordnet')




def abstracter(n_articles,query,fields = "journal,abstract,title,year",fieldsofstudy = ""):

    headers = {"x-api-key": 'zS5ZuEKrRZ6PZF3GHOTErudmnAclsFN37gYvaOkd'}
    requests.get("https://api.semanticscholar.org/", headers = headers)
    journals = []
    abstracts = []
    titles = []
    years = []
    n_articles = n_articles
    baseurl = "https://api.semanticscholar.org/graph/v1/paper/search?query="
    query = query
    rest = "&limit=100&fields="
    fields = fields
    #"publicationTypes,journal,title,abstract,year"
    s2FieldsOfStudy = "&s2FieldsOfStudy="
    fieldsofstudy = fieldsofstudy
    typer = query+" "+fieldsofstudy


    for i in range(0,n_articles//100):
        offset = "&offset="+f"{i*100}"
        if fieldsofstudy == "":
            request = baseurl+query+offset+rest+fields
        request = baseurl+query+offset+rest+fields+s2FieldsOfStudy+fieldsofstudy

        df = requests.get(request, headers = headers)
        print(df)
        time.sleep(2)
        for i in range(0,100):
            try:
                journals.append(df.json()["data"][i]["journal"]["name"])
                titles.append(df.json()["data"][i]["title"])
                abstracts.append(df.json()["data"][i]["abstract"])
                years.append(df.json()["data"][i]["year"])
                
            except:
                next
    return(pd.DataFrame({
	'journals': journals,
	'abstracts': abstracts,
	'years': years,
	'title': titles,
    "type": typer}))


def journals(data,min_val):

    filter_value = min_val
    dictionary = Counter(data["journals"])
    filterddic = {k: v for k, v in dictionary.items() if v > filter_value}
    plt.bar(list(filterddic.keys()), filterddic.values(), color='g')
    plt.xticks(rotation=90)
    plt.show()


def cleaner(data, year = None):
    if year != None:
        data = data[data['years'] > year]

    data = data.dropna()
    data[data['abstracts'].str.strip().astype(bool)]
    data[data['title'].str.strip().astype(bool)]
    
    data["abstracts"] = data["abstracts"].str.lower()
    data["title"] = data["title"].str.lower()

    
    return(data)




def get_umap(text, sentencetransformer = 'all-MiniLM-L6-v2', dim = 2, typer = " "):
    model = SentenceTransformer(sentencetransformer)
    embeddings = model.encode(text)
    umapp = umap.UMAP(n_components=dim)
    proj = umapp.fit_transform(embeddings)
    if dim <= 3:    
        if dim == 3:
            fig = px.scatter_3d(
            proj, x=0, y=1, z=2,
            color=typer,labels={'color': 'Research area'}
            )
        else:
            fig = px.scatter(
                proj, x=0, y=1,
                color=typer, labels={'color': 'Research area'}
                )
        fig.update_layout(template="simple_white")
        fig.show()

    if dim > 3:
        print("cannot visualize over 3D")
    return(proj)

def determin_clustersize(proj, cluster_size = [5,10,20,50,100,200]):
    for cluters in cluster_size:
        cluster = hdbscan.HDBSCAN(min_cluster_size=cluters)
        cluster.fit(proj)
        cluster.condensed_tree_.plot(select_clusters=True)
        plt.title(f"Cluster size {cluters}")
        plt.show()


def vis_clustersize(proj, min_cluster = 10):
    plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
    palette = sns.color_palette()
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
    cluster.fit(proj)
    cluster_colors = [sns.desaturate(palette[col], sat)
                    if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                    zip(cluster.labels_, cluster.probabilities_)]

    plt.scatter(proj.T[0], proj.T[1], c=cluster_colors, **plot_kwds)



def fitter(abstracts,umap_dim,min_cluster,embed_model = 'all-MiniLM-L6-v2',stopwords = True,top_n_words=10):
    
    vectorizer_model = CountVectorizer(stop_words="english")
    umap_model = umap.UMAP(n_components=umap_dim)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
    embed_model = SentenceTransformer(embed_model)

    if stopwords == False:

        topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embed_model,
        top_n_words=top_n_words,
        language = "english")

        topics, probs = topic_model.fit_transform(abstracts)
        return(topics, probs, topic_model)



    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embed_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        language = "english")

    topics, probs = topic_model.fit_transform(abstracts)
    return(topics, probs, topic_model)


#LDA analysis:
def lemmatize_stemming(text):
    lemmatized = WordNetLemmatizer().lemmatize(text, pos='v')
    return PorterStemmer().stem(lemmatized)


def preprocess(text):
    result = []
    #gensim does tokenization and makes everything lowercase.
    for token in gensim.utils.simple_preprocess(text):
        #want to remove the stopwords and tokens that have fewer than 3 characters
        if token not in stopwords.words("english"):
            #append the resulting tokens after being lemmatized and stemmed
            result.append(lemmatize_stemming(token))
    return result

def get_corpus(text):
    processed_docs = text.map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    return(dictionary,bow_corpus,tfidf_corpus)


def get_weights(model, corpus):
    topic_weights = []
    for row_list in model[corpus]:
        weight_list = []
        for i, w in row_list:
            weight_list.append(w)
        topic_weights.append(weight_list)
    arr = pd.DataFrame(topic_weights).fillna(0).values
    return(arr)


def barchart_lda(lda_model,num_topics):
    num_topics = num_topics
    
    columns = 4
    rows = int(np.ceil(num_topics / columns))
    width = 250
    height = 250

    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.2,
                        subplot_titles = [f"Topic {topic}" for topic in range(num_topics)])

    row = 1
    column = 1

    for i in range(num_topics):
        fig.add_trace(go.Bar(x=[height for _,height in lda_model.show_topic(i)[::-1]],
                        y = [x for x,_ in lda_model.show_topic(i)[::-1] ],
                        orientation='h'), row = row, col = column)
        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"))