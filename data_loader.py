import twitter
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from config import *
from datetime import date, datetime, time
import pytz

class DB(object):
    def __init__(self,config,db_name,autoconnect=True):
        self.config = config
        self.db_name = db_name
        if autoconnect:
            self.connect()

    def connect(self):
        self.client = MongoClient(**self.config)
        self.db = self.client[self.db_name]

    def put(self, coll_name, item):
        item['_id']=item['id']
        del item['id']
        try:
            id = self.db[coll_name].insert_one(item).inserted_id
        except DuplicateKeyError as er:
            print("dublicate id {}".format(item['_id']))
            return None
        return id  

    def get_collection(self,coll_name,start_date=None):

        if start_date:
            res = list(self.db[coll_name].find({'created_at': {'$gte': start_date}}))
            return res
        return list(self.db[coll_name].find())

    def get_max_id(self, coll_name):
        if coll_name not in self.db.collection_names():return None
        col = self.db[coll_name]
        return col.find_one(sort=[("_id", -1)]).get('_id')




api = twitter.Api(consumer_key='8Ua6ZZjTZAKmQN3hrgxDmMRWx',
                      consumer_secret='Hkg4Sdd9Q1yzuhQY3j2i4KXzMd70iVRLt6p0g3PBemA7sakABh',
                      access_token_key='2254796784-zvSyhgXDZ9sMp9G9IFw3sTZPVhlf6Rfsq7p0fYr',
                      access_token_secret='tjuiGjjJnBDafAV4YgPd9dzIExWVM4NsqzX3oJB1moV4O')

dbclient = DB(config_mongo, db_namespace)

def load_twitter_data(sources,count,dbclient):
    for source in sources:
        print('load ' ,source)
        tweets = api.GetUserTimeline(screen_name=source,count=count)
        for tweet in tweets:
            js = tweet._json
            js['created_at'] = datetime.strptime(js['created_at'],'%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.UTC)
            dbclient.put(source,js)

def update(sources,count,dbclient):
    for source in sources:
        print('update ' ,source)
        max_id = dbclient.get_max_id(source)
        if not max_id:
            print('this source not exist in db ', source)
            print('try to load ', source)
            load_twitter_data([source],200,dbclient)
            continue
        tweets = api.GetUserTimeline(screen_name=source,count=count,since_id=max_id)
        print('new tweets ', len(tweets))
        for tweet in tweets:
            js = tweet._json
            js['created_at'] = datetime.strptime(js['created_at'],'%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.UTC)
            dbclient.put(source,js)


def create_dataset(dbclient,collections,fields,start_date=None):
    list_dataset = []
    for col in collections: 
        source_data = dbclient.get_collection(col,start_date)
        dict_tweets = []
        for tweet in source_data:
            dict_tweet = {}
            dict_tweet['source'] = col
            for f in fields:
                dict_tweet[f] = tweet[f]

            dict_tweets.append(dict_tweet)

        list_dataset.extend(dict_tweets)

    return list_dataset

