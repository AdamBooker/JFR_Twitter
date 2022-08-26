import requests 
import json
import pandas as pd
from math import ceil
import time
from datetime import timedelta

CLIENT_ID = "Add your client ID here"
SECRET_KEY = "Add your secret key here"

# create authentication object
auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_KEY)
reddit_headers = {'User-Agent': 'research test for u/Iskiicyblithely'}

data = {
    'grant_type': 'password',
    'username': 'Add your user name here', # enter own reddit username
    'password': 'Add your password here' # enter own reddit password associated with the username
}

token = requests.request('POST', 'https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=reddit_headers).json()['access_token']
reddit_headers['Authorization']= f'bearer {token}'

def reddit_oauth(r):
    r.headers = reddit_headers
    return r


def connect_to_endpoint_reddit(url, auth, headers, rate_limit):
    begin_time = time.time()
    # for calculating wait times
    rate_limit_window = timedelta(minutes=rate_limit)
    response = requests.request('GET', url, auth=auth,headers=headers)
    if response.status_code == 429:
        sleepy_time = rate_limit_window - timedelta(seconds=ceil(time.time()-begin_time))
        time.sleep(sleepy_time.seconds+10)

        #try again
        begin_time = time.time()
        return connect_to_endpoint_reddit(url,auth,headers,rate_limit)
    elif response.status_code != 200:
        raise Exception(f'Error reddit: {response.status_code} {response.text}')
    
    return response.json()


response = connect_to_endpoint_reddit('https://oauth.reddit.com/r/wallstreetbets/comments', reddit_oauth, reddit_headers,5)

with open('out/reddit_wallstreetbets.json','w') as f:
    for i in response['data']['children']:
        f.write(json.dumps(i)+'\n')

    while True:
        if 'data' in response:
            if 'after' in response['data']:
                headers = {'before': response['data']['after']}
                response = connect_to_endpoint_reddit('https://oauth.reddit.com/r/wallstreetbets/comments', reddit_oauth, reddit_headers,5)
                for i in response['data']['children']:
                    f.write(json.dumps(i)+'\n')
                # REMOVE THE break DIRECTLY BELOW TO PARSE POSTS BEFORE THIS SET
                break
            else:
                break
        else:
            break