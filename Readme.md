# requirements

* Installed mongo. I use dockerized mongo https://hub.docker.com/_/mongo/. After instal configure connection for mongo in config.py
* run pip install requirements.txt

# usage

* start web app : python app.py
* GET request on http://0.0.0.0:8000/clusters_report  with curl or paste in a browser for generating the report about clusters with similar tweets
* GET request on http://0.0.0.0:8000/<year-month-day> . example: http://0.0.0.0:8000/hotest_tweets/2016-12-25.
Result will be file with two most hottest tweets from date what you typed in the url

