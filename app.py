from io import BytesIO

from flask import Flask
from flask import send_file

from analytical_utils import init_dataset,make_clusters,select_best_tweet,fields
from data_loader import *
from config import *
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
import time 
import pandas as pd
from datetime import datetime,date, time as dtime
from werkzeug.routing import BaseConverter, ValidationError
from gensim.models import Word2Vec


class DateConverter(BaseConverter):
    """Extracts a ISO8601 date from the path and validates it."""

    regex = r'\d{4}-\d{2}-\d{2}'

    def to_python(self, value):
        try:
            return datetime.strptime(value, '%Y-%m-%d').date()
        except ValueError:
            raise ValidationError()

    def to_url(self, value):
        return value.strftime('%Y-%m-%d')

unique_tweets = None
all_clusters = None
wv = None

app = Flask(__name__)
app.url_map.converters['date'] = DateConverter


@app.route("/hotest_tweets/<from_date>", methods=['GET'])
def hello(from_date):    
    best_two = select_best_tweet(unique_tweets,all_clusters,from_date)
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    best_two.to_excel(writer, startrow = 0, merge_cells = False, sheet_name = "Sheet_1")
    writer.close()
    output.seek(0)
    return send_file(output, attachment_filename="best_two.xlsx", as_attachment=True)


@app.route("/clusters_report", methods=['GET'])
def cluters_report():
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    all_clusters.to_excel(writer, startrow = 0, merge_cells = False, sheet_name = "Sheet_1")
    writer.close()
    output.seek(0)
    return send_file(output, attachment_filename="report.xlsx", as_attachment=True)


def init_wv():
    wv = Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",
        binary=True)
    wv.init_sims(replace=True)
    return wv

def crone_task():
    # start_date = date.today()
    # start_datetime = datetime.combine(start_date, time.min)

    print('start update data from twitter')
    update(twitter_user_list,200,dbclient)
    print('start init clusters')
    df = init_dataset()
    global unique_tweets,all_clusters
    unique_tweets,all_clusters = make_clusters(df,wv)
    print('unique_tweets length ', len(unique_tweets))
    print('all_clusters length ', len(all_clusters))


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.start()
    scheduler.add_job(
    func=crone_task,
    trigger=IntervalTrigger(seconds=600),
    id='update',
    name='update dat',
    replace_existing=True)
    atexit.register(lambda: scheduler.shutdown())

    print('load word2vec')
    global wv
    start = time.time()
    wv = init_wv()
    end = time.time()
    print('init wv after ', end - start)

    start = time.time()
    crone_task()
    end = time.time()
    print('init clusters after ', end - start)
    print('init complete')
    app.run(port=8000,host='0.0.0.0')