# coding: utf-8
import pandas as pd
import numpy as np
from config import *
from data_loader import create_dataset,dbclient
from nltk.corpus import stopwords
import nltk
import gensim
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from datetime import date

fields = ['created_at','favorite_count', 'favorited','retweet_count','retweeted','text']

def init_dataset(start_date=None):
    df = pd.DataFrame(create_dataset(dbclient,twitter_user_list,fields,start_date))
    df.created_at = pd.to_datetime(df.created_at)
    df['url'] = df.text.str.extract("(?P<url>https?://[^\s]+)")
    df.text = df.text.str.replace("(?P<url>https?://[^\s]+)", '')
    # priori assumptions: retweet_count multiplying by 3 because it retweet more sagnificant factor about "hotness" of event.
    df['fav_and_retweet'] = df.favorite_count + df.retweet_count * 3
    return df

def make_clusters(df,wv):
    
    text_cluster = df.groupby('text').filter(lambda x: x['text'].count() > 1)
    
    text_clusters = []
    for t in text_cluster.text.unique():
        dict_cluster = {}
        tempdf = text_cluster[text_cluster.text == t]
        dict_cluster['all_urls'] = '; '.join(tempdf.url.tolist())
        dict_cluster['url_with_max_favorite'] = tempdf.url.loc[np.argmax(tempdf.favorite_count)]
        dict_cluster['url_with_max_retweet'] = tempdf.url.loc[np.argmax(tempdf.retweet_count)]
        dict_cluster['max_favorite_count'] = tempdf.favorite_count.max()
        dict_cluster['retweet_count'] = tempdf.retweet_count.max()
        dict_cluster['url_with_max_fav_and_retweet'] = tempdf.url.loc[np.argmax(tempdf.fav_and_retweet)]
        dict_cluster['fav_and_retweet_count'] = tempdf.fav_and_retweet.max()
        dict_cluster['first_date'] = tempdf.created_at.min()
        dict_cluster['text'] = tempdf.text.values[0]
        dict_cluster['cluster_size'] = len(tempdf)
        text_clusters.append(dict_cluster)    
    

    unique_by_text = df.groupby('text').filter(lambda x: x['text'].count() == 1)
    url_cluster = unique_by_text.groupby('url').filter(lambda x: x['url'].count() > 1)
    url_clusters = []
    for t in url_cluster.url.unique():
        dict_cluster = {}
        tempdf = url_cluster[url_cluster.url == t]
        dict_cluster['all_urls'] = '; '.join(tempdf.url.tolist())
        dict_cluster['url_with_max_favorite'] = tempdf.url.loc[np.argmax(tempdf.favorite_count)]
        dict_cluster['url_with_max_retweet'] = tempdf.url.loc[np.argmax(tempdf.retweet_count)]
        dict_cluster['max_favorite_count'] = tempdf.favorite_count.max()
        dict_cluster['retweet_count'] = tempdf.retweet_count.max()
        dict_cluster['url_with_max_fav_and_retweet'] = tempdf.url.loc[np.argmax(tempdf.fav_and_retweet)]
        dict_cluster['fav_and_retweet_count'] = tempdf.fav_and_retweet.max()
        dict_cluster['first_date'] = tempdf.created_at.min()
        dict_cluster['text'] = tempdf.text.values[0]
        dict_cluster['cluster_size'] = len(tempdf)
        url_clusters.append(dict_cluster) 


    unique_by_text_and_url = unique_by_text.groupby('url').filter(lambda x: x['url'].count() == 1)

    url_clusters_df = pd.DataFrame(url_clusters)
    text_clusters_df = pd.DataFrame(text_clusters)
    
    tokenized = unique_by_text_and_url['text'].apply(lambda r: w2v_tokenize_text(r)).values
    
    wv_cluster_df,wv_indexes = make_clusters_with_vw(wv, unique_by_text_and_url, tokenized)

    text_clusters_df['clustered_by'] = 'text'
    url_clusters_df['clustered_by'] = 'url'
    wv_cluster_df['clustered_by'] = 'wv_with_cos_sym_0.97'
    all_clusters = pd.concat([text_clusters_df,url_clusters_df,wv_cluster_df])
    unique_indexes = list(set(wv_indexes).symmetric_difference(set(range(len(unique_by_text_and_url)))))
    unique_tweets = unique_by_text_and_url.iloc[unique_indexes]

    return unique_tweets, all_clusters


def make_clusters_with_vw(wv,unique_by_text_and_url,tokenized):
    word_average = word_averaging_list(wv,tokenized)
    
    sym_matrix = cosine_similarity(word_average)
    
    sym_indexes = np.column_stack(np.where((sym_matrix < 0.9999999999) & (sym_matrix > 0.97)))
    
    edges = dict({frozenset(tup) for tup in sym_indexes.tolist()})
    
    graph = SimpleGraph()
    graph.edges = edges
    
    wv_cluster_indexes = search(graph)

    wv_cluster = []
    wv_indexes = []
    for ind in wv_cluster_indexes:
        wv_indexes.extend(ind)
        dict_cluster = {}
        tempdf = unique_by_text_and_url.iloc[ind]
        dict_cluster['all_urls'] = '; '.join(tempdf.url.tolist())
        dict_cluster['url_with_max_favorite'] = tempdf.url.loc[np.argmax(tempdf.favorite_count)]
        dict_cluster['url_with_max_retweet'] = tempdf.url.loc[np.argmax(tempdf.retweet_count)]
        dict_cluster['max_favorite_count'] = tempdf.favorite_count.max()
        dict_cluster['retweet_count'] = tempdf.retweet_count.max()
        dict_cluster['url_with_max_fav_and_retweet'] = tempdf.url.loc[np.argmax(tempdf.fav_and_retweet)]
        dict_cluster['fav_and_retweet_count'] = tempdf.fav_and_retweet.max()
        dict_cluster['first_date'] = tempdf.created_at.min()
        dict_cluster['text'] = tempdf.text.values[0]
        dict_cluster['cluster_size'] = len(tempdf)
        wv_cluster.append(dict_cluster)  

    wv_cluster_df = pd.DataFrame(wv_cluster)
    return wv_cluster_df,wv_indexes

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if word in stopwords.words('english'):
                continue
            tokens.append(word)
    return tokens

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        print("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list ])


class SimpleGraph:
    def __init__(self):
        self.edges = {}
    
    def neighbors(self, id):
        return self.edges.get(id)

def search(graph):
    visited = {}
    clusters = []
    for current in graph.edges.keys():
        visited[current] = True
        next = graph.neighbors(current)
        cluster = [current]

        while next != None and next not in visited:
            cluster.append(next)
            visited[next] = True
            next = graph.neighbors(next)
            
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters
        

def select_best_tweet(unique_tweets,all_clusters,start_date):
    all_today_clusters = all_clusters[all_clusters.first_date >= start_date]
    unique_today_tweets = unique_tweets[unique_tweets.created_at >= start_date]

    all_today_clusters.rename( columns={"url_with_max_fav_and_retweet": "url",
                                        "fav_and_retweet_count": "fav_and_retweet",
                                       "max_favorite":"favorite_count",
                                       "max_retweet":"retweet_count"}, inplace=True)
    
    
    concated = pd.concat((unique_today_tweets,all_today_clusters))
    return concated.loc[concated.fav_and_retweet.nlargest(2).index][['url','text','favorite_count','retweet_count']]


