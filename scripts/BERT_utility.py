
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
from plotly.subplots import make_subplots
from distinctipy import distinctipy
from typing import List
import plotly.io as pio
import random
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


def get_umap(data, analysis, sentencetransformer = 'all-MiniLM-L6-v2', dim = 2, random=True):
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
    if random == True:
        umapp = umap.UMAP(n_components=dim)
    else:
        umapp = umap.UMAP(n_components=dim, random_state=24)
    model = SentenceTransformer(sentencetransformer)
    #model embeddings and fitting UMAP
    embeddings = model.encode(data[analysis].tolist())
    proj = umapp.fit_transform(embeddings)

    #plotting
    if dim <= 3:    
        if dim == 3:
            fig = px.scatter_3d(
            proj, x=0, y=1, z=2,
            color=data["type"],labels={'color': 'Research area'}
            )
        else:
            fig = px.scatter(
                proj, x=0, y=1,
                color=data["type"], labels={'color': 'Research area'}
                )
        fig.update_layout(template="simple_white")
        fig.show()

    if dim > 3:
        print("cannot visualize over 3D")
    return(proj)


def determin_clustersize(proj, cluster_size = [5,10,15,20]):
    """Clustering algorithm HDBSCAN, takes projections and clusteres them, size of clusters are determined by cluster_size
        Returns:
        Figure of the condensed_treeplot from HDBSCAN to visualise clustering steps from different clustersizes and a colored scatterplot of identified clusters.
    """
    plt.figure(figsize=(8, 6), dpi=80)
    for cluters in cluster_size:
        cluster = hdbscan.HDBSCAN(min_cluster_size=cluters)
        cluster.fit(proj)   
        plt.figure()
        cluster.condensed_tree_.plot(select_clusters=True)
        plt.title(f"Cluster size {cluters}")
        plt.show()
        try:
            palette  = distinctipy.get_colors(35)
            #first has to be grey (which is the onces the clustering algorithm calls -1)
            random.shuffle(palette)
            if len(np.unique(np.array(cluster.labels_))) < 35:
                cluster_colors = [palette[col] for col in cluster.labels_]
                plt.figure()
                plt.scatter(proj.T[0], proj.T[1], c=cluster_colors)

        except:
            print("Try increasing the cluster_size, there are over 50 clusters")




def fitter(data, analysis,umap_dim,min_cluster,embed_model = 'all-MiniLM-L6-v2',stopwords = True,top_n_words=10, random = True):
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

    
    if random == True:
        umap_model = umap.UMAP(n_components=umap_dim)
    else:
        umap_model = umap.UMAP(n_components=umap_dim, random_state=24)
    
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
    embed_model = SentenceTransformer(embed_model)
    embeddings = embed_model.encode(data[analysis].tolist())

    if stopwords == False:
        topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=top_n_words,
        language = "english")
        topics, probs = topic_model.fit_transform(data[analysis].tolist(),embeddings)
        return(topics, probs, topic_model)
    else:
        vectorizer_model = CountVectorizer(stop_words="english")
        
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=top_n_words,
            language = "english")

        topics, probs = topic_model.fit_transform(data[analysis].tolist(),embeddings)
        return(topics, probs, topic_model)



def barchart_bert(topic_model,num_topics,analysis):
    """Function to visualize the BERTopic-model topic results.
        Returns:
        Barchart of top 5 words in the number of topics specified.
    """

    pio.renderers.default = 'png'
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
    height = 300
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
        font=dict(size=28),
        hoverlabel=dict(
        bgcolor="white",
        font_family="Rockwell"))
    return fig


def run_bert(data, analysis = "abstracts", save_plot = True, file = "BERT_run", clustersize = 15, random = True):
    """Wrapper function to run the BERTopic analysis 
        Returns:
        topics, probabilities of topics and the topic model
    """
    
    topics, probs, topic_model = fitter(data, analysis = analysis,umap_dim = 2,min_cluster = clustersize, random=random)
    fig = barchart_bert(topic_model=topic_model, num_topics=3, analysis = analysis)

    if save_plot == True:
        if not os.path.exists("BERTopic_results"):
            os.mkdir("BERTopic_results")
            
        fig.write_image(os.path.join(os.getcwd(),f"BERTopic_results/{file}_analysis={analysis}.png"), engine = "auto")
        
    return(topics, probs, topic_model)


def run_explorative(data,analysis = "abstracts", save_plot = True, clustersize = 22, random = False):
    topics, probs, topic_model = fitter(data, analysis = analysis, umap_dim = 2, min_cluster = clustersize,random = random)
    
    fig1 = topic_model.visualize_barchart(top_n_topics = 16, n_words = 3)
    if save_plot == True:
        if not os.path.exists("BERTopic_Psychedelics"):
            os.mkdir("BERTopic_Psychedelics")
        fig1.write_image(os.path.join(os.getcwd(),f"BERTopic_Psychedelics/barchart_analysis={analysis}.png"), engine = "auto")

        topics_over_time = topic_model.topics_over_time(data[analysis], data["years"])
        if analysis == "abstracts":
            fig = topic_model.visualize_topics_over_time(topics_over_time, topics=[0,1,2,5,7,13,15])
            fig.update_layout(font=dict(size=16))
            fig.write_image(os.path.join(os.getcwd(),f"BERTopic_Psychedelics/topics_overtime_analysis={analysis}.png"), engine = "auto")
        else:
            fig = topic_model.visualize_topics_over_time(topics_over_time, topics=[1,2,4,7,11,12])
            fig.update_layout(font=dict(size=16))
            fig.write_image(os.path.join(os.getcwd(),f"BERTopic_Psychedelics/topics_overtime_analysis={analysis}.png"), engine = "auto")


