import requests
import numpy as np
import pandas as pd

from google.cloud import bigquery
from google.cloud import language_v1


from pprint import pprint


from urllib.request import urlopen
from bs4 import BeautifulSoup

import concurrent.futures
import multiprocessing
from tqdm import tqdm

from ratelimit import limits, sleep_and_retry

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

nlp_client = language_v1.LanguageServiceClient()

def _analyze_sentiments(text:str):
    """
    Given a text, return a "sentiments" object that encapsulates the overall and sentence-wise sentiments of the text. 
    Args:
        text (str): a string representing a news article
        
    Returns:
        sentiments (language_service.AnalyzeSentimentResponse)
    """
    document = language_v1.Document(
        content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language = "en"
    )
    sentiments = nlp_client.analyze_sentiment(
        request={"document": document}
    )
    return sentiments


def _extract_persons(text: str):
    """
    Given a text, return a list of all noteworthy people mentioned in the text
    The presence of a Wikipedia page is used as proxy of "noteworthiness"
    
    Args:
        text (str): a string representing a news article
        
    Returns:
        persons (list of language_service.Entity) 
    """
    document = language_v1.Document(
        content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language = "en"
    )
    entities = nlp_client.analyze_entities(
        request={"document": document}
    )
    # filter for persons
    persons = [entity for entity in entities.entities if entity.type_.name == 'PERSON']
    # filter for noteworthy persons using presence of a Wikipedia page as proxy
    persons = [person for person in persons if ("wikipedia_url" in person.metadata)]
    return persons


def _get_en_url(org_url_link: str):
    """
    Given a non-English Wikipedia URL, return the English version
    
    Args:
        org_url_link (str): the original Wikipedia URL (non English)
    Returns:
        en_url_link (str): the English URL
    """
    soup = BeautifulSoup(urlopen(org_url_link), features="html.parser")

    interlanguage_links = soup.select('li.interlanguage-link > a')
    try:
        en_url_link = [el.get("href") for el in interlanguage_links if el.get('lang') == 'en'][0]
        return en_url_link
    except:
        raise IndexError("No English equivalent exists for the given link. Maybe the link is already in English?")
        
def _get_title_and_en_url(org_url_link: str):
    """
    Given a Wikipedia URL, get its English version (if it's not English already) and the English article title
    Args:
        org_url_link (str): a Wikipedia URL
    Returns a tuple of:
        - title: the English title of the Wikipedia page
        - en_url_link (str): the English URL
    """
    # 
    if org_url_link[8:10] == "en":
        en_url_link = org_url_link
    else:
        en_url_link = _get_en_url(org_url_link)
    soup_en = BeautifulSoup(urlopen(en_url_link), features="html.parser")
    title = soup_en.find(id="firstHeading").contents[0]
    return (title, en_url_link)


# 250 calls per minute
CALLS = 250
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def analyze_article(text: str, postprocess: bool = False): 
    """
    Given a text, analyze it by getting its overall sentiment and extracting the noteworthy individuals mentioned within the text and their details.
    Args:
        - text (str): the textual content of a news article
        - postprocess: if yes, convert the Wikipedia URI to an English version and change the person's name to how it's written in the Wikipedia page
    Returns a tuple of:
        - article_sentiment (float): the overall sentiment of the article, as determined by Natural Language API
        - persons (list of dict): a list of 'person' dictionary. This dictionary contains the following keys:
            - name
            - wikipedia_uri
            - num_setence (number of sentences in the article in which the person is mentioned)
            - person_sentiment: the average sentiment of all sentences in which the person is mentioned
    """
    # API call
    sentiments = _analyze_sentiments(text)
    persons = _extract_persons(text)
    
    # Extraction of results
    article_sentiment = sentiments.document_sentiment.score
    person_dicts = []
    for person in persons:
        person_dict = {}
        org_url_link = person.metadata["wikipedia_url"]
        if postprocess:
            try:
                # some non-latin characters in the original URL might raise an error
                person_dict["name"], person_dict["wikipedia_uri"] = _get_title_and_en_url(org_url_link)
            except:
                pass
        else:
            person_dict["name"] = person.name
            person_dict["wikipedia_uri"] = org_url_link
        
        # for each sentence in which the person's name is mentioned, get its sentiment
        sentence_sentiments = [sentence_obj.sentiment.score for sentence_obj in sentiments.sentences if person.name in sentence_obj.text.content]
        person_dict["person_sentiment"] = np.mean(sentence_sentiments)
        person_dict["num_sentences"] = len(sentence_sentiments)

        person_dicts.append(person_dict)

    return article_sentiment, person_dicts

def draw_boxplot_histogram(df: pd.DataFrame, col: str) -> None:
    """ 
    Given a dataframe `df` containing a numeric column `col`, plot the distribution of `col` using a histogram and a boxplot
    """
    f, (ax_box, ax_hist) = plt.subplots(2, figsize = (18,6), sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(x = df[col], ax=ax_box)
    sns.histplot(data=df, x=col, ax=ax_hist, stat = "percent", kde = True)
    ax_hist.axvline(x=df[col].median(),
            color='red')
    ax_box.set(xlabel='')
    ax_hist.set(xlabel=f"{col} (median: {df[col].median():,.1f})")
    plt.show()