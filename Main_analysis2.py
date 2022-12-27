from utility_funtions import *
from LDA_utility import *
from BERT_utility import *
import random
import argparse

def main():
    np.random.seed(0)
    random.seed(0)

    data = get_data(queries = ["Visual Perception", "Auditory perception","Pain perception"],n_articles=2000, year = 1990,key = 'zS5ZuEKrRZ6PZF3GHOTErudmnAclsFN37gYvaOkd', fields = "journal,abstract,title,year")
 
    run_bert(data, analysis = "abstracts", save_plot = True, file = "Visual_Auditory_Pain", clustersize = 20,randomstate = 42)
    run_bert(data, analysis = "title", save_plot = True, file = "Visual_Auditory_Pain", clustersize = 20,randomstate = 42)

    run_LDA(data, num_topics = 3, analysis = "abstracts", save_plot = True, file = "Visual_Auditory_Pain", bow = True, alpha = "auto",random_state = 42)
    run_LDA(data, num_topics = 3, analysis = "title", save_plot = True, file = "Visual_Auditory_Pain", bow = True, alpha = "auto",random_state = 42)


