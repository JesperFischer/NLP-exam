from BERT_utility import *
from utility_funtions import *
import random


def main_3(query,n_articles,year,key,clustersize):
    np.random.seed(0)
    random.seed(0)    

    #get data
    data3 = get_data(queries = [query],n_articles=n_articles, year = year,key = key, fields = "journal,abstract,title,year",fieldsofstudy = "Medicine,Biology")
    #run modeling
    
    topics, probs, topic_model = run_bert(data3, analysis = "abstracts", save_plot = False, clustersize = clustersize,randomstate = 42)
    fig1 = topic_model.visualize_barchart(top_n_topics = 16, n_words = 3)
    if not os.path.exists("BERTopic_Psychedelics"):
        os.mkdir("BERTopic_Psychedelics")
    fig1.write_image(os.path.join(os.getcwd(),f"BERTopic_Psychedelics/barchart.png"), engine = "auto")


    topics_over_time = topic_model.topics_over_time(data3["abstracts"], data3["years"])
    fig = topic_model.visualize_topics_over_time(topics_over_time, topics=[2,3,4,5,12,13])
    fig.update_layout(font=dict(size=16))
    fig.write_image(os.path.join(os.getcwd(),f"BERTopic_Psychedelics/topics_overtime.png"), engine = "auto")

    return(topic_model)




def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--query', type = str, default = "Psychedelics")
    parser.add_argument("-n", "--n_articles", type=int, default=2000)
    parser.add_argument("-y", "--year", type=int, default=1990)
    parser.add_argument("-k", "--key", type=str, default="zS5ZuEKrRZ6PZF3GHOTErudmnAclsFN37gYvaOkd")
    parser.add_argument("-c", "--cluster_size", type=int, default=15)

    # Parse arguments
    args = parser.parse_args()

    return args



if __name__ == "__main3__":
    args = parseArguments()
    main_3(query=args.query, 
        n_articles=args.n_articles, 
        year=args.year,
        key=args.key, 
        cluster_size=args.cluster_size)