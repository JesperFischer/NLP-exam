"""
Contains functions for loading and cleaning data.
"""
import time
import pandas as pd
import requests
from nltk.corpus import stopwords
from typing import List


def abstracter(n_articles : int, query : str,key : str, fields = "journal,abstract,title,year",fieldsofstudy = "") -> pd.DataFrame:
    """loads scientific abstracts and other information from semantic scholar
    Args:
        n_articles (int): number of articles extracted
        query (str): Your desired keyword search
        key (str): API key given from semantic scholar
        fields (str): information extracted from semantic scholar, see shorturl.at/bsxZ1 for options
        fieldsofstudy (str): limit search to particular field of study options are found in the link: shorturl.at/bsxZ1
    Returns:
        Dataframe: A dataframe containing the specified fields.
    """


    #specify api key:
    headers = {"x-api-key": key}
    requests.get("https://api.semanticscholar.org/", headers = headers)
    #empty lists to store data
    journals = []
    abstracts = []
    titles = []
    years = []
    n_articles = n_articles
    baseurl = "https://api.semanticscholar.org/graph/v1/paper/search?query="
    query = query
    rest = "&limit=100&fields="
    fields = fields
    #"publicationTypes,journal,title,abstract,year"
    s2FieldsOfStudy = "&s2FieldsOfStudy="
    fieldsofstudy = fieldsofstudy
    typer = query+" "+fieldsofstudy

    #looping through the articles need to take 100 at a time as that is what is recommeneded on semantic scholar
    for i in range(0,n_articles//100):
        offset = "&offset="+f"{i*100}"
        #getting the right address:
        if fieldsofstudy == "":
            request = baseurl+query+offset+rest+fields
        request = baseurl+query+offset+rest+fields+s2FieldsOfStudy+fieldsofstudy
        #getting the abstracts
        df = requests.get(request, headers = headers)
        #sleep to make sure it doesn't get overloaded
        time.sleep(2)
        #append the data to the empty lists
        for i in range(0,100):
            try:
                journals.append(df.json()["data"][i]["journal"]["name"])
                titles.append(df.json()["data"][i]["title"])
                abstracts.append(df.json()["data"][i]["abstract"])
                years.append(df.json()["data"][i]["year"])
                #
            except:
                next
    #return a dataframe with the filled lists
    return(pd.DataFrame({
	'journals': journals,
	'abstracts': abstracts,
	'years': years,
	'title': titles,
    "type": typer}))


def cleaner(data : pd.DataFrame, year = None) -> pd.DataFrame:
    """cleans and if year is given filters dataframe from abstracter function
    Returns:
        Dataframe: A dataframe containing the specified fields.
    """
    #removing articles that are below the year specified
    if year != None:
        data = data[data['years'] > year]
    #dropping NA values
    data = data.dropna()
    #getting rid of articles that have no title or abstract
    data[data['abstracts'].str.strip().astype(bool)]
    data[data['title'].str.strip().astype(bool)]
    #convert all abstracts and titles to lower
    data["abstracts"] = data["abstracts"].str.lower()
    data["title"] = data["title"].str.lower()
    return(data)

def get_data(queries: str,n_articles:int,year : int, key:str, fields = "journal,abstract,title,year", fieldsofstudy = "") -> pd.DataFrame:
    """wrapper function of cleaning and getting data
    Args:
        n_articles (int): number of articles extracted
        query (str): Your desired keyword search
        key (str): API key given from semantic scholar
        fields (str): information extracted from semantic scholar, see shorturl.at/bsxZ1 for options
        fieldsofstudy (str): limit search to particular field of study options are found in the link: shorturl.at/bsxZ1
    Returns:
        Dataframe: A dataframe containing the specified fields.
    """
    queries = queries.split(", ")
    data = pd.DataFrame()
    for i in queries:
        data1 = abstracter(n_articles=n_articles,query = i, key = key, fields=fields, fieldsofstudy = "")
        data1 = cleaner(data1, year = year)
        data = pd.concat([data,data1])
    return(data)