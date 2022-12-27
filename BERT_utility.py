
"""
Contains functions for fitting and visualizing BERTopic results.
"""
import plotly.express as px
import umap
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import hdbscan
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from nltk.corpus import stopwords
import os
import plotly.graph_objects as go
import numpy as np
import nltk
from plotly.subplots import make_subplots
from typing import List
import random
nltk.download('wordnet')


def get_umap(text, sentencetransformer = 'all-MiniLM-L6-v2', dim = 2, typer = " "):
    """Uniform Manifold Approximation and Projection demensionality reduction technique, coverts texts / documents to word embeddings and reduces these embeddings to 2 or 3D space and plots them
    Args:
        text (list[str]): text to be embedded and demensionality reduced, in a list format
        sentencetransformer (str): The sentencetransformer to be used to make the word embeddings.
        dim (int): number of demensions the embeddings should be reduced to
        typer (str): optional to color the word embeddings in the resulting scatterplot (can be used with typer from abstracter function)
    Returns:
        projections of the word embeddings and a scatter plot if dim < 3.
    """
    #defining models
    umapp = umap.UMAP(n_components=dim)
    model = SentenceTransformer(sentencetransformer)
    #model embeddings and fitting UMAP
    embeddings = model.encode(text)
    proj = umapp.fit_transform(embeddings)

    #plotting
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


def determin_clustersize(proj, cluster_size = [5,10,15,20], as_scatter = True):
    """Clustering algorithm HDBSCAN, takes projections and clusteres them, size of clusters are determined by cluster_size
        Returns:
        Figure of the condensed_treeplot from HDBSCAN to visualise clustering steps from different clustersizes and a colored scatterplot of identified clusters.
    """
    for cluters in cluster_size:
        cluster = hdbscan.HDBSCAN(min_cluster_size=cluters)
        cluster.fit(proj)   
        plt.figure()
        cluster.condensed_tree_.plot(select_clusters=True)
        plt.title(f"Cluster size {cluters}")
        plt.show()
        if as_scatter == True:
            plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
            palette = sns.color_palette("hls", 50)
            random.shuffle(palette)
            if len(np.unique(np.array(cluster.labels_))) < 50:
                cluster_colors = [sns.desaturate(palette[col], sat)
                                if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                                zip(cluster.labels_, cluster.probabilities_)]

            else:
                print("Try increasing the cluster_size, there are over 50 clusters")
            plt.figure()
            plt.scatter(proj.T[0], proj.T[1], c=cluster_colors, **plot_kwds)

            

def fitter(text,umap_dim,min_cluster,embed_model = 'all-MiniLM-L6-v2',stopwords = True,top_n_words=10, randomstate = 42):
    """Function to Fit a BERTopic model with userdefined inputs
    Args:
        text (list[str]): text to be topicmodelled, in a list format
        umap_dim (int): number of demensions the word embeddings are reduced to before clustering
        min_cluster(int): cluster_size should be determined by visual inspection from determin_clustersize
        sentencetransformer (str): The sentencetransformer to be used to make the word embeddings.
        stopwords (Logical): Should stopwords be removed?
        top_n_words (int): How many words per topic should be returned?
    Returns:
        projections of the word embeddings and a scatter plot if dim < 3.
    """
    vectorizer_model = CountVectorizer(stop_words="english")
    umap_model = umap.UMAP(n_components=umap_dim, random_state=randomstate)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
    embed_model = SentenceTransformer(embed_model)

    if stopwords == False:
        topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embed_model,
        top_n_words=top_n_words,
        language = "english")

        topics, probs = topic_model.fit_transform(text)
        return(topics, probs, topic_model)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embed_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        language = "english")

    topics, probs = topic_model.fit_transform(text)
    return(topics, probs, topic_model)



def barchart_bert(topic_model,num_topics,analysis):
    """Function to visualize the BERTopic-model topic results.
        Returns:
        Barchart of top 5 words in the number of topics specified.
    """
    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    topics = sorted(freq_df.Topic.to_list()[:num_topics])
    # Initialize figure
    
    
    columns = 3
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.2,
                        vertical_spacing=.4,
                        subplot_titles=[f"Topic {topic}" for topic in topics])

    # Add barchart for each topic
    rows = int(np.ceil(num_topics / columns))
    width = 250
    height = 250
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:5][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:5][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h'),
            row=row, col=column)
        fig.update_xaxes(tickangle=90)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        title = f"Topics on {analysis}",
        template="plotly_white",
        showlegend=False,
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        font=dict(size=24),
        hoverlabel=dict(
        bgcolor="white",
        font_family="Rockwell"))
    return fig


def run_bert(data, analysis = "abstracts", save_plot = True, file = "BERT_run", clustersize = 15, randomstate = 42):
    """Wrapper function to run the BERTopic analysis 
        Returns:
        topics, probabilities of topics and the topic model
    """
    
    topics, probs, topic_model = fitter(data[analysis].tolist(),umap_dim = 2,min_cluster = clustersize, randomstate=randomstate)
    fig = barchart_bert(topic_model=topic_model, num_topics=3, analysis = analysis)

    if save_plot == True:
        if not os.path.exists("BERTopic_results"):
            os.mkdir("BERTopic_results")
            
        fig.write_image(os.path.join(os.getcwd(),f"BERTopic_results/{file}_analysis={analysis}.png"), engine = "auto")
        
    return(topics, probs, topic_model)