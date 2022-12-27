
# Topic modeling on scientific abstracts and titles

This is the repo used for the paper,
"Topic modeling on scientific abstracts and titles".
For complete reproducibility of the plots and results in the paper
follow the steps in "Running the code".





## Running the code

To rerun the whole analysis from the paper simply type the following code in the termial:
This code will install all requirements and use the data gathered from semantic scholar inside the data-folder. 

```bash
bash run.sh
```

### Running other experiments
To run other experiments than the one presented in the paper, users have to visit https://www.semanticscholar.org/ and request an API key
which has to be entered into the key argument in the get_data function.


### importing data
importing abstracts or titles are performed by the function "get_data"
So to extract 1000 abstracts and titles from the year 2000 until today with the keywords:
Cardiology, Perception and Natural Language Processing. Simply write:

```python
from utility_funtions import *
#insert API-key here:
key = ""

data = get_data(queries = "Cardiology, Perception, Natural Language Processing",
                key = key
                n_articles = 1000,
                year = 2000)
```

## Determining hyperparameters for BERTopic
To find sensible hyperparameters for BERTopic for either abstracts or titles simply get the 2D projections
form UMAP using the following code (this is for abstracts)
To make the code reproducible include random = False
```python
from BERT_utility import *

proj = get_umap(data["abstracts"].tolist(), typer = data["type"], random = False)
```


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


##

To determine the cluster size that fits with what one expects after investigating the above plot simply type:

```python
from BERT_utility import *

determin_clustersize(proj, cluster_size = [5,10,15,20,25])
```


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


To run the whole BERTopic that uses cluster-based term freqency inverse document freqency (c-tf-idf) simply do: 



```python 
topic, prob, topic_model = fitter(data["abstracts"].tolist(),umap_dim = 2,min_cluster = 20,embed_model = 'all-MiniLM-L6-v2',stopwords = True,random = False)
```

## Visualizing results
Now to visualize the results as an interactive intertopic distance map, the following code can be run

```python 
topic_model.visualize_topics()
```

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


### To get the dynamic topic representation:

```python 

topics_over_time = topic_model.topics_over_time(data["abstracts"], data["years"])
topic_model.visualize_topics_over_time(topics_over_time)
```

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

