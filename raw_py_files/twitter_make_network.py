import requests 
import json
import os
import pandas as pd
from math import ceil
import re
import time
from datetime import timedelta
import codecs
import networkx as nx


def grouper(group_size,infile_path,data_col):

        '''
        Make a list of lists of csvs with length group_size

        group_size: batch size of groups
        infile_path: path to Excel file with input accounts
        users_col: header of column with Twitter user names
        '''
        # Declare file path and group size
        # See Twitter API docs for max group size related to your call
        df = pd.read_excel(infile_path)
        
        # get list of accounts from DataFrame
        # remove @ signs and spaces
        accounts = [re.sub('[@ ]', '', z) for z in df[data_col].to_list()]
        
        # Parse to list of groups
        num_groups = ceil(len(accounts)/group_size)
        groups = []
        for i in range(num_groups): groups.append(','.join(accounts[i*group_size:(i+1)*(group_size)]))
        return groups


bearer_token = 'Add API token here'
def oauth(r):
    '''API Authorization Callback'''
    r.headers['Authorization'] = f'Bearer {bearer_token}'
    r.headers['User-Agent'] = 'v2UserTweetsPython'
    return r


def connect_to_endpoint(url, params, rate_limit):
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
        response = requests.request('GET', url=url, auth=oauth, params=params)
        # handle rate limit
        if response.status_code == 429:
            sleepy_time = rate_limit_window - timedelta(seconds=ceil(time.time()-begin_time))
            time.sleep(sleepy_time.seconds+10)

            #try again
            begin_time = time.time()
            return connect_to_endpoint(params, begin_time)

        elif response.status_code != 200:
            raise Exception(f'Error: {response.status_code} {response.text}')
        return response.json()


# number of accounts in each request
# see API docs for limits
group_size = 20

# location of excel file with user accounts
in_file = 'in/ceo_sample.xlsx'

# column header with user accounts in in_file
user_column = 'Twitter Account'

#Twitter v2 API endpoint 
api_endpoint = 'https://api.twitter.com/2/users/by'

# number of minutes in rate limit
rate_limit = 15

# use the grouping code in twitter_rest
groups = grouper(group_size,in_file,user_column)
out_file = 'in/user_ids.json'

for k,users in enumerate(groups):
    params = {}
    params['usernames'] = users
    params['user.fields'] = 'verified,id,public_metrics,username'
    resp = connect_to_endpoint(api_endpoint,params,rate_limit)

    #write response to file
    if k == 0:
        m = 'w'
    else:
        m = 'a'
    with open(out_file, m) as f:
        for r in resp['data']:
            f.write(json.dumps(r)+'\n')


url = 'https://api.twitter.com/2/users/{id}/tweets'

with open('in/user_ids.json') as f:
    out_data = []

    for k,j in enumerate(f):
        user = json.loads(j)
        api_endpoint = url.format(id=user['id'])
        params = {}
        params['expansions'] = 'in_reply_to_user_id'
        params['max_results'] = 100
        
        resp = connect_to_endpoint(api_endpoint, params, rate_limit)
        try:
            while 'next_token' in resp['meta']:
                params['pagination_token'] = resp['meta']['next_token']
                resp = connect_to_endpoint(api_endpoint, params, rate_limit)
                for k,z in enumerate(resp['data']):
                    if 'in_reply_to_user_id' in z:
                        uid = user['id']
                        reply_uid = z['in_reply_to_user_id']
                        tweet_id = z['id']
                        text = codecs.encode(re.sub('\n', ' ', z['text']), 'utf-8',errors='ignore')
                        out_data.append([str(uid),str(reply_uid)])
        except:
            # here if user deleted their account
            continue
        
with open('out/edge_list3200.txt', 'w') as f:
    f.write('uid\treply_uid\n')
    for i in out_data:
        f.write('\t'.join(i)+'\n',)


with open('out/edge_list3200.txt', 'rb') as f:
    # remove header
    next(f)

    # add source and target from edge list
    g = nx.read_edgelist(f, delimiter='\t')
print(g)
nx.draw(g, pos=nx.kamada_kawai_layout(g))