import requests 
import json
import pandas as pd
from math import ceil
import time
from datetime import timedelta

BEARER_TOKEN = 'Enter your API key here'
def twitter_oauth(r):
    '''API Authorization Callback'''
    r.headers['Authorization'] = f'Bearer {BEARER_TOKEN}'
    r.headers['User-Agent'] = 'Pycathalon Nopebooks, HAL 5000, & 42'
    return r


def connect_to_endpoint_twitter(url, params, rate_limit, auth):
    '''
    API Connector

    rate_limit: rate limit in minutes
    url: API endpoint
    params: Query Parameters
    begin_time: unix time of query start
    '''
    begin_time = time.time()
    # for calculating wait times
    rate_limit_window = timedelta(minutes=rate_limit)


    # url, auth=bearer_oauth, params=params
    response = requests.request('GET', url=url, auth=auth, params=params)
    # handle rate limit
    if response.status_code == 429:
        sleepy_time = rate_limit_window - timedelta(seconds=ceil(time.time()-begin_time))
        time.sleep(sleepy_time.seconds+10)

        #try again
        begin_time = time.time()
        return connect_to_endpoint_twitter(url, params, rate_limit, auth)

    elif response.status_code != 200:
        raise Exception(f'Error twitter: {response.status_code} {response.text}')
    return response.json()


# Rules tell the Twitter server which Tweets to return. 
# See the current set of rules at this endpoint
url = 'https://api.twitter.com/2/tweets/search/stream/rules'
response = connect_to_endpoint_twitter(url, None, 1, auth=twitter_oauth)
rules = response

# Delete previous rules
url = 'https://api.twitter.com/2/tweets/search/stream/rules'
if rules is not None and 'data' in rules:
    payload = {'delete': {'ids': [z['id'] for z in rules['data']]}}
    sc = requests.request('POST', url, auth=twitter_oauth,json=payload).status_code
    if sc != 200:
        print(f'Rule deletion error status code: {sc}')

# Make a rule to get all posts with the cashtag $TSLA
# Making a rule requires sending data to Twitter with a Post request. 
# The function connect_to_endpoint() works with Get requests, so we will use another request.
# We could add a "GET", "Post" option to connect_to_endpoint() using the request below
# https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule#types

rule = {'add': [{'value':'$TSLA'}]}
# Notice the term 'POST' is used to send data (the data is the rule in this case)
requests.request('POST', url, auth=twitter_oauth, json=rule)


url = 'https://api.twitter.com/2/tweets/search/stream'
# Here there is another new term in our requests: stream. 
# The boolean variable stream is used to keep the connection to twitter open.
response = requests.request('GET', url, auth=twitter_oauth, stream=True)

# This will keep adding posts to the file below until there is an exception
# Remove break to make this code continuously monitor Twitter
with open('out/twitter_tsla.json', 'wb') as f:
    for r in response.iter_lines():
        if r:
            f.write(r+b'\n')
            break










