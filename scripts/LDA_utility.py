"""
Contains functions for extra cleaning of data as well as fitting and visualizing LDA results.
"""

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from plotly.subplots import make_subplots
import gensim
import os
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')



#LDA analysis:
def lemmatize_stemming(text):
    """Function to lemmatize a text or string
        Returns:
        Returns the lemmatized text
    """
    lemmatized = WordNetLemmatizer().lemmatize(text, pos='v')
    return PorterStemmer().stem(lemmatized)


def preprocess(text):
    """Function to preprocess (lemmatize, remove stopwords & make everything lowercase) a text
        Returns:
        Returns the preprocessed text
    """
    result = []
    #gensim does tokenization and makes everything lowercase.
    for token in gensim.utils.simple_preprocess(text):
        #want to remove the stopwords and tokens that have fewer than 3 characters
        if token not in stopwords.words("english"):
            #append the resulting tokens after being lemmatized and stemmed
            result.append(lemmatize_stemming(token))
    return result


def get_corpus(text):
    """Function to that takes the preprocessed texts and returns a dicitionary with a tfidf and Bag of words representation
        Returns:
        dictionary, bag of words representation and tf-idf representation of the processed text
    """
    processed_docs = text.map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tfidf = gensim.models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    return(dictionary,bow_corpus,tfidf_corpus)

def barchart_lda(lda_model,num_topics, analysis):
    """Function to visualize the LDA-model topic results.
        Returns:
        Barchart of top 5 words in each topic
    """
    pio.renderers.default = 'png'
    num_topics = num_topics
    columns = 3
    rows = int(np.ceil(num_topics / columns))
    width = 250
    height = 250

    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.2,
                        vertical_spacing=.4,
                        subplot_titles = [f"Topic {topic}" for topic in range(num_topics)])

    row = 1
    column = 1
    for i in range(num_topics):
        fig.add_trace(go.Bar(x=[height for _,height in lda_model.show_topic(i,topn = 5)[::-1]],
                        y = [x for x,_ in lda_model.show_topic(i,topn = 5)[::-1] ],
                        orientation='h'), row = row, col = column)
        fig.update_xaxes(tickangle=90)
        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

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
    fig.show()
    return(fig)



def run_LDA(data, num_topics, analysis = "abstracts", save_plot = True, file = "LDA_run", bow = 1, alpha = "auto", random = True):
    """Function to Fit a LDA model with user defined inputs
    Args:
        data (Dataframe): dataframe containing a column that should be used for LDA-topic modelings (colum defined in analysis)
        num_topics (int): number of topics to be extracted from the analysis
        analysis (str): column in dataframe to be used for the topic model, column should include rows as documents
        save_plot (Logical): should the function make a folder and save the top 5 words in each of the topics?
        file(str): file name if save plot is true
        bow (Logical): use of a bag of words representation if True if False uses a tf-idf representation
        alpha (str; numpy_array; list of floats): prior belief of topic distribution in documents, see https://radimrehurek.com/gensim/models/ldamodel.html for options
    Returns:
        Barchart of top 5 words in each topic and the LDA topic model
    """
    dictionary,bow_corpus,tfidf_corpus = get_corpus(data[analysis])
    #fitting
    if random != True:
        np.random.RandomState(24)
    if bow == 1:
        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, alpha=alpha, eval_every=5)
        analysis_type = "bow"
    else:
        lda_model = gensim.models.ldamodel.LdaModel(tfidf_corpus, num_topics=num_topics, id2word=dictionary, alpha=alpha, eval_every=5)
        analysis_type = "tf-idf"
   #plotting
    fig = barchart_lda(lda_model, num_topics, analysis)

    if save_plot == True:
        if not os.path.exists("LDA_results"):
            os.mkdir("LDA_results")
            
        fig.write_image(os.path.join(os.getcwd(),f"LDA_results/{file}_analysis={analysis}_type={analysis_type}.png"), engine = "auto")
        
    return(fig, lda_model)


