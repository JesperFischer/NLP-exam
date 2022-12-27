from utility_funtions import *
from LDA_utility import *
from BERT_utility import *
import random
import argparse

def main(query, n_articles, year,key,cluster_size):
    queries = query.split(", ")
    np.random.seed(0)
    random.seed(0)

    data = get_data(queries = queries,n_articles=n_articles, year = year,key = key, fields = "journal,abstract,title,year")

    #run BERT
    run_bert(data, analysis = "abstracts", save_plot = True, file = "NLP_Perception_Cardiology", clustersize = cluster_size,randomstate = 42)
    run_bert(data, analysis = "title", save_plot = True, file = "NLP_Perception_Cardiology", clustersize = cluster_size,randomstate = 42)

    #run LDA
    run_LDA(data, num_topics = len(queries), analysis = "abstracts", save_plot = True, file = "NLP_Perception_Cardiology", bow = True, alpha = "auto")
    run_LDA(data, num_topics = len(queries), analysis = "title",save_plot = True, file = "NLP_Perception_Cardiology", bow = True, alpha = "auto")


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--query', type = str, default = "Natural Language Processing, Perception, Cardiology")
    parser.add_argument("-n", "--n_articles", type=int, default=2000)
    parser.add_argument("-y", "--year", type=int, default=1990)
    parser.add_argument("-k", "--key", type=str, default="zS5ZuEKrRZ6PZF3GHOTErudmnAclsFN37gYvaOkd")
    parser.add_argument("-c", "--cluster_size", type=int, default=15)

    # Parse arguments
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parseArguments()
    main(query=args.query, 
        n_articles=args.n_articles, 
        year=args.year,
        key=args.key, 
        cluster_size=args.cluster_size)