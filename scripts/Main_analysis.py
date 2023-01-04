import pandas as pd
from LDA_utility import *
from BERT_utility import *
import argparse

def main(appendix):
    """Runs the main analysis of the NLP-exam and reproduces the plots provided in the paper
    Appendix (int): if set to 1 plots and analyses from the appendix is run.
    """
    #getting the data
    data1 = pd.read_csv(os.path.join(os.getcwd(),"data","NLP_perception_cardiology.csv"))
    data2 = pd.read_csv(os.path.join(os.getcwd(),"data","Visual_Auditory_Pain_perception.csv"))
    data3 = pd.read_csv(os.path.join(os.getcwd(),"data","Psychedelics.csv"))
    
    #run appendix analysis (tf-idf on LDA)?
    if appendix == 1:
        run_LDA(data1, num_topics = 3, analysis = "abstracts", save_plot = True, file = "NLP_perception_cardiology", bow = 0, alpha = "auto", random = False)
        run_LDA(data1, num_topics = 3, analysis = "title",save_plot = True, file = "NLP_perception_cardiology", bow = 0, alpha = "auto", random = False)
        #run LDA on second proof-of-concept
        run_LDA(data2, num_topics = 3, analysis = "abstracts", save_plot = True, file = "Visual_Auditory_Pain_perception", bow = 0, alpha = "auto", random = False)
        run_LDA(data2, num_topics = 3, analysis = "title",save_plot = True, file = "Visual_Auditory_Pain_perception", bow = 0, alpha = "auto", random = False)

        run_bert(data1, analysis = "abstracts", file = "NLP_perception_cardiology", tfidf = 1, clustersize = 15,random = False)
        run_bert(data1, analysis = "title", file = "NLP_perception_cardiology", tfidf = 1, clustersize = 15,random = False)
        #run BERT on second proof-of-concept
        run_bert(data2, analysis = "abstracts", file = "Visual_Auditory_Pain_perception", tfidf = 1, clustersize = 15,random = False)
        run_bert(data2, analysis = "title", file = "Visual_Auditory_Pain_perception", tfidf = 1, clustersize = 15,random = False)
        

    #run LDA on first proof-of-concept

    run_LDA(data1, num_topics = 3, analysis = "abstracts", save_plot = True, file = "NLP_perception_cardiology", bow = 1, alpha = "auto", random = False)
    run_LDA(data1, num_topics = 3, analysis = "title",save_plot = True, file = "NLP_perception_cardiology", bow = 1, alpha = "auto", random = False)
    #run LDA on second proof-of-concept
    run_LDA(data2, num_topics = 3, analysis = "abstracts", save_plot = True, file = "Visual_Auditory_Pain_perception", bow = 1, alpha = "auto", random = False)
    run_LDA(data2, num_topics = 3, analysis = "title",save_plot = True, file = "Visual_Auditory_Pain_perception", bow = 1, alpha = "auto", random = False)


    #run BERT on first proof-of-concept
    run_bert(data1, analysis = "abstracts", file = "NLP_perception_cardiology", tfidf = 0, clustersize = 18,random = False)
    run_bert(data1, analysis = "title", file = "NLP_perception_cardiology", tfidf = 0, clustersize = 18,random = False)
    #run BERT on second proof-of-concept
    run_bert(data2, analysis = "abstracts", file = "Visual_Auditory_Pain_perception", tfidf = 0, clustersize = 22,random = False)
    run_bert(data2, analysis = "title", file = "Visual_Auditory_Pain_perception", tfidf = 0, clustersize = 22,random = False)
    

    #run last analysis with BERTopic:
    run_explorative(data3,analysis = "abstracts", clustersize = 11, random = False)
    run_explorative(data3,analysis = "title", clustersize = 22, random = False)
    

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    #appendix option
    parser.add_argument('-a','--appendix', type = int, default = 1)
    
    # Parse arguments
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parseArguments()
    main(
        appendix=args.appendix
        )