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
