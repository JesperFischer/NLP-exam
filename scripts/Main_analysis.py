import pandas as pd
from LDA_utility import *
from BERT_utility import *
import argparse

def main(cluster_size, bow):

    data1 = pd.read_csv(os.path.join(os.getcwd(),"data","NLP_perception_cardiology.csv"))
    data2 = pd.read_csv(os.path.join(os.getcwd(),"data","Visual_Auditory_Pain_perception.csv"))
    

    #run LDA on first proof-of-concept

    run_LDA(data1, num_topics = 3, analysis = "abstracts", save_plot = True, file = "NLP_perception_cardiology", bow = bow, alpha = "auto", random = False)
    run_LDA(data1, num_topics = 3, analysis = "title",save_plot = True, file = "NLP_perception_cardiology", bow = bow, alpha = "auto", random = False)
    #run LDA on second proof-of-concept
    run_LDA(data2, num_topics = 3, analysis = "abstracts", save_plot = True, file = "Visual_Auditory_Pain_perception", bow = bow, alpha = "auto", random = False)
    run_LDA(data2, num_topics = 3, analysis = "title",save_plot = True, file = "Visual_Auditory_Pain_perception", bow = bow, alpha = "auto", random = False)



    #run BERT on first proof-of-concept
    run_bert(data1, analysis = "abstracts", save_plot = True, file = "NLP_perception_cardiology", clustersize = cluster_size,random = False)
    run_bert(data1, analysis = "title", save_plot = True, file = "NLP_perception_cardiology", clustersize = cluster_size,random = False)
    #run BERT on second proof-of-concept
    run_bert(data2, analysis = "abstracts", save_plot = True, file = "Visual_Auditory_Pain_perception", clustersize = cluster_size,random = False)
    run_bert(data2, analysis = "title", save_plot = True, file = "Visual_Auditory_Pain_perception", clustersize = cluster_size,random = False)
    

    #run last analysis with BERTopic:
    run_explorative(data3,analysis = "abstracts", clustersize = 22, random = False)
    run_explorative(data3,analysis = "title", clustersize = 22, random = False)
    

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--bow', type = int, default = 1)
    parser.add_argument("-c", "--cluster_size", type=int, default=20)
    # Parse arguments
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parseArguments()
    main(
        cluster_size=args.cluster_size,
        bow=args.bow,
        )