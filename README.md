
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


## importing data
importing abstracts or titles are performed by the function "get_data"
So to extract 500 abstracts and titles (per search word) from the year 2000 until today with the keywords specified in queries. 

```python
from utility_funtions import *
#insert API-key here:
key = ""

data = get_data(queries = "Homeostasis, Perception, Brain, Natural Language Processing, linguistics",
                key = key,
                n_articles = 500,
                year = 2000)

```

## Determining hyperparameters for BERTopic
To find sensible hyperparameters for BERTopic for either abstracts or titles simply get the 2D projections
form UMAP using the following code (this is for abstracts)
To make the code reproducible include random = False
```python
from BERT_utility import *

proj = get_umap(data, analysis = "abstracts", random = False)
```


![](Readme_figures/UMAP.png)


To determine the cluster size use the function determin_clustersize, which takes a list of cluster_sizes as arguments and plot the resulting clustering.

```python
from BERT_utility import *

determin_clustersize(proj, cluster_size = [16])
```


![](Readme_figures/Condenced_cluster.png)           ![](Readme_figures/Implication_of_condenced_cluster.png)


To run the whole BERTopic that uses combines these steps with cluster-based term freqency inverse document freqency (c tf-idf).

```python 
topic, prob, Bertopic_model = fitter(data,
                                    analysis = "abstracts",
                                    umap_dim = 2,
                                    min_cluster = 13,
                                    stopwords = True,
                                    random = False)
```

### Visualizing the topics
Now to visualize the results as an interactive intertopic distance map, the following code can be run

```python 
Bertopic_model.visualize_barchart()
```

![](Readme_figures/Topics.png)


### To get the dynamic topic representation:

```python 

topics_over_time = Bertopic_model.topics_over_time(data["abstracts"], data["years"])
Bertopic_model.visualize_topics_over_time(topics_over_time)
```

![](Readme_figures/Topics_over_time.png)

## Running other experiments with LDA
To run topic modeling on the same data using the LDA framework:

```python 
fig, lda_model = run_LDA(data, 10, analysis = "abstracts", save_plot = False, bow = 1, alpha = "auto", random = True):
```

![](Readme_figures/LDA_topics.png)